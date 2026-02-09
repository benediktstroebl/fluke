"""ColBERTv2 baseline: standard late interaction model with MaxSim scoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TokenEncoder
from ..scoring.maxsim import maxsim


class ColBERTModel(nn.Module):
    """ColBERTv2-style late interaction model.

    Encodes queries and documents independently into per-token embeddings,
    then scores relevance using MaxSim: for each query token, find max
    similarity to any doc token, then sum.

    This serves as the baseline for comparison with FLUKE.
    """

    def __init__(
        self,
        model_name: str = "small",
        embedding_dim: int = 128,
        query_max_length: int = 32,
        doc_max_length: int = 180,
    ):
        super().__init__()
        self.encoder = TokenEncoder(model_name, embedding_dim)
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.embedding_dim = embedding_dim
        # Special query token markers ([Q] token approach from ColBERT)
        # We prepend [unused0] as [Q] marker for queries
        self.query_marker_token_id = self.encoder.tokenizer.convert_tokens_to_ids(
            "[unused0]"
        )
        # Use [unused1] as [D] marker for documents
        self.doc_marker_token_id = self.encoder.tokenizer.convert_tokens_to_ids(
            "[unused1]"
        )

    def encode_queries(
        self, queries: list[str], batch_size: int = 32
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode queries into per-token embeddings.

        Returns list of (embeddings, mask) tuples.
        """
        all_results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            tokens = self.encoder.tokenize(batch, max_length=self.query_max_length)
            tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
            with torch.no_grad():
                embs, masks = self.encoder(tokens["input_ids"], tokens["attention_mask"])
            for j in range(embs.shape[0]):
                mask = masks[j]
                all_results.append((embs[j][mask].cpu(), mask[mask].cpu()))
        return all_results

    def encode_documents(
        self, documents: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode documents into per-token embeddings.

        Returns list of (embeddings, mask) tuples.
        """
        from tqdm import tqdm

        all_results = []
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents", total=len(documents) // batch_size + 1)

        for i in iterator:
            batch = documents[i : i + batch_size]
            tokens = self.encoder.tokenize(batch, max_length=self.doc_max_length)
            tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
            with torch.no_grad():
                embs, masks = self.encoder(tokens["input_ids"], tokens["attention_mask"])
            for j in range(embs.shape[0]):
                mask = masks[j]
                all_results.append((embs[j][mask].cpu(), mask[mask].cpu()))
        return all_results

    def score(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a single query-document pair using MaxSim."""
        return maxsim(query_embs, doc_embs, query_mask, doc_mask)

    def score_batch(
        self,
        query_embs: torch.Tensor,
        doc_embs_list: list[torch.Tensor],
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one query against multiple documents."""
        scores = []
        for doc_embs in doc_embs_list:
            scores.append(self.score(query_embs, doc_embs, query_mask))
        return torch.stack(scores)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training: encode and score query-document pairs.

        Args:
            query_input_ids: (batch, q_len)
            query_attention_mask: (batch, q_len)
            doc_input_ids: (batch, d_len)
            doc_attention_mask: (batch, d_len)

        Returns:
            (batch,) relevance scores
        """
        q_embs, q_mask = self.encoder(query_input_ids, query_attention_mask)
        d_embs, d_mask = self.encoder(doc_input_ids, doc_attention_mask)

        scores = []
        for i in range(q_embs.shape[0]):
            s = maxsim(q_embs[i], d_embs[i], q_mask[i], d_mask[i])
            scores.append(s)
        return torch.stack(scores)
