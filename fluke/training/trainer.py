"""Training loop for late interaction models (ColBERT and FLUKE).

Uses contrastive learning with in-batch negatives, following the standard
ColBERTv2 training recipe: for each (query, positive_doc) pair, treat all
other in-batch documents as negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TripletDataset(Dataset):
    """Dataset of (query, positive_doc, negative_doc) triplets."""

    def __init__(self, triplets: list[tuple[str, str, str]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def collate_triplets(batch, tokenizer, query_max_length=32, doc_max_length=180):
    """Collate triplets into tokenized batches."""
    queries, pos_docs, neg_docs = zip(*batch)
    q_tok = tokenizer(
        list(queries), padding=True, truncation=True,
        max_length=query_max_length, return_tensors="pt",
    )
    p_tok = tokenizer(
        list(pos_docs), padding=True, truncation=True,
        max_length=doc_max_length, return_tensors="pt",
    )
    n_tok = tokenizer(
        list(neg_docs), padding=True, truncation=True,
        max_length=doc_max_length, return_tensors="pt",
    )
    return q_tok, p_tok, n_tok


def train_epoch(model, dataloader, optimizer, device="cpu"):
    """Train for one epoch using pairwise contrastive loss + in-batch negatives.

    Training uses two losses:
    1. Triplet margin loss: uses model.forward() (full scoring including all
       components) so all parameters get gradients.
    2. In-batch negative cross-entropy: uses CQI-weighted MaxSim for clean
       encoder gradients. The scoring innovations (ASC, MGS, TIR) get their
       gradients from the triplet loss, while the encoder gets strong gradients
       from both losses.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for q_tok, p_tok, n_tok in tqdm(dataloader, desc="Training"):
        q_ids = q_tok["input_ids"].to(device)
        q_mask = q_tok["attention_mask"].to(device)
        p_ids = p_tok["input_ids"].to(device)
        p_mask = p_tok["attention_mask"].to(device)
        n_ids = n_tok["input_ids"].to(device)
        n_mask = n_tok["attention_mask"].to(device)

        # Full model forward pass: all components get gradients
        pos_scores = model(q_ids, q_mask, p_ids, p_mask)
        neg_scores = model(q_ids, q_mask, n_ids, n_mask)

        # Pairwise margin ranking loss
        loss = F.relu(1.0 - pos_scores + neg_scores).mean()

        # In-batch negatives: CQI-weighted MaxSim for clean encoder gradients
        batch_size = q_ids.shape[0]
        if batch_size > 1:
            q_embs, q_masks = model.encoder(q_ids, q_mask)
            d_embs, d_masks = model.encoder(p_ids, p_mask)

            # CQI importance if available
            if hasattr(model, "cqi") and model.cqi is not None:
                importances = model.cqi(q_embs, q_masks)
            else:
                importances = q_masks.float()

            all_scores = []
            for i in range(batch_size):
                row_scores = []
                for j in range(batch_size):
                    # CQI-weighted MaxSim (fast, clean gradients for encoder+CQI)
                    from ..scoring.fluke_scoring import importance_weighted_maxsim
                    ws, _ = importance_weighted_maxsim(
                        q_embs[i], d_embs[j], importances[i],
                        q_masks[i], d_masks[j],
                    )
                    row_scores.append(ws.sum())
                all_scores.append(torch.stack(row_scores))
            score_matrix = torch.stack(all_scores)

            labels = torch.arange(batch_size, device=device)
            ib_loss = F.cross_entropy(score_matrix, labels)
            loss = loss + ib_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_model(
    model,
    triplets: list[tuple[str, str, str]],
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-6,
    device: str = "cpu",
):
    """Full training loop."""
    dataset = TripletDataset(triplets)
    tokenizer = model.encoder.tokenizer

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_triplets(
            b, tokenizer,
            query_max_length=model.query_max_length if hasattr(model, "query_max_length") else 32,
            doc_max_length=model.doc_max_length if hasattr(model, "doc_max_length") else 180,
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.to(device)
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}")

    return model
