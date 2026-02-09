"""FLUKE: Fused Late-interaction with Unified Key Embeddings.

A novel late interaction retrieval model that improves upon ColBERTv2 through:

1. **Contextual Query Importance (CQI)**: A lightweight attention head that
   produces per-query-token importance weights from query context. Important
   terms (rare, specific, semantically rich) contribute more to the score.

2. **Soft Top-K Aggregation**: Instead of hard max over document tokens per
   query token, aggregate top-K matches with softmax weighting. This is more
   robust to spurious high-similarity matches.

3. **Token Interaction Residual (TIR)**: A small MLP that takes the vector
   of per-query-token match scores and produces a residual correction,
   capturing cross-term dependencies that MaxSim fundamentally cannot
   (e.g., "both terms A and B must match together").

All document representations are identical to ColBERTv2 (pre-computed token
embeddings), so FLUKE preserves the same indexing efficiency. The additional
components operate only at query time and add negligible compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TokenEncoder
from ..scoring.fluke_scoring import (
    fluke_score,
    importance_weighted_maxsim,
    TokenInteractionResidual,
)


class ContextualQueryImportance(nn.Module):
    """Produces per-query-token importance weights from query context.

    Uses the [CLS] token's representation to attend over query tokens and
    produce a scalar importance weight for each. This captures:
    - Token specificity (rare terms should matter more)
    - Contextual importance (same term has different importance in different queries)
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.attn_proj = nn.Linear(embedding_dim, embedding_dim)
        self.importance_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Initialize near-uniform so we start close to ColBERTv2
        for layer in self.importance_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self, query_embeddings: torch.Tensor, query_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute importance weights for each query token.

        Args:
            query_embeddings: (batch, num_tokens, dim) or (num_tokens, dim)
            query_mask: (batch, num_tokens) or (num_tokens,)

        Returns:
            importance_weights: same shape as query_mask, positive weights
        """
        single = query_embeddings.dim() == 2
        if single:
            query_embeddings = query_embeddings.unsqueeze(0)
            query_mask = query_mask.unsqueeze(0)

        # Use CLS token (index 0) as context
        cls_token = query_embeddings[:, 0:1, :]  # (batch, 1, dim)
        projected = self.attn_proj(cls_token)  # (batch, 1, dim)

        # Attention-like scoring over all query tokens
        attn_scores = (projected * query_embeddings).sum(dim=-1)  # (batch, num_tokens)

        # Also use per-token self-importance
        token_importance = self.importance_head(query_embeddings).squeeze(-1)

        # Combine attention-based and token-based importance
        raw_weights = attn_scores + token_importance

        # Mask and normalize
        raw_weights = raw_weights.masked_fill(~query_mask, float("-inf"))
        # Softmax normalization, then scale by number of valid tokens
        # so total weight ≈ num_valid_tokens (preserving scale with ColBERTv2)
        num_valid = query_mask.float().sum(dim=-1, keepdim=True)
        weights = F.softmax(raw_weights, dim=-1) * num_valid

        if single:
            weights = weights.squeeze(0)

        return weights


class FLUKEModel(nn.Module):
    """FLUKE: Fused Late-interaction with Unified Key Embeddings.

    Extends ColBERTv2 with three innovations for better retrieval quality
    while maintaining the same indexing efficiency.
    """

    def __init__(
        self,
        model_name: str = "small",
        embedding_dim: int = 128,
        query_max_length: int = 32,
        doc_max_length: int = 180,
        topk: int = 3,
        temperature: float = 0.1,
        use_tir: bool = True,
        use_cqi: bool = True,
        use_soft_topk: bool = True,
    ):
        super().__init__()
        self.encoder = TokenEncoder(model_name, embedding_dim)
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.embedding_dim = embedding_dim
        self.topk = topk
        self.temperature = temperature
        self.use_tir = use_tir
        self.use_cqi = use_cqi
        self.use_soft_topk = use_soft_topk

        # Innovation 1: Contextual Query Importance
        if use_cqi:
            self.cqi = ContextualQueryImportance(embedding_dim)
        else:
            self.cqi = None

        # Innovation 3: Token Interaction Residual
        if use_tir:
            self.tir = TokenInteractionResidual(
                max_query_tokens=query_max_length, hidden_dim=64
            )
        else:
            self.tir = None

    def encode_queries(
        self, queries: list[str], batch_size: int = 32
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode queries into per-token embeddings with importance weights.

        Returns list of (embeddings, mask, importance_weights) tuples.
        """
        all_results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            tokens = self.encoder.tokenize(batch, max_length=self.query_max_length)
            tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
            with torch.no_grad():
                embs, masks = self.encoder(tokens["input_ids"], tokens["attention_mask"])
                if self.cqi is not None:
                    weights = self.cqi(embs, masks)
                else:
                    weights = masks.float()
            for j in range(embs.shape[0]):
                mask = masks[j]
                w = weights[j][mask]
                all_results.append((embs[j][mask].cpu(), mask[mask].cpu(), w.cpu()))
        return all_results

    def encode_documents(
        self, documents: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode documents into per-token embeddings.

        Document encoding is identical to ColBERTv2 — the innovations are
        all on the query/scoring side. This preserves indexing efficiency.
        """
        from tqdm import tqdm

        all_results = []
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator, desc="Encoding documents",
                total=len(documents) // batch_size + 1,
            )

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
        importance_weights: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a single query-document pair using FLUKE scoring."""
        topk = self.topk if self.use_soft_topk else None
        return fluke_score(
            query_embs, doc_embs, importance_weights,
            tir_module=self.tir,
            query_mask=query_mask, doc_mask=doc_mask,
            topk=topk, temperature=self.temperature,
            max_query_tokens=self.query_max_length,
        )

    def score_batch(
        self,
        query_embs: torch.Tensor,
        doc_embs_list: list[torch.Tensor],
        importance_weights: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one query against multiple documents."""
        scores = []
        for doc_embs in doc_embs_list:
            scores.append(self.score(query_embs, doc_embs, importance_weights, query_mask))
        return torch.stack(scores)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training."""
        q_embs, q_mask = self.encoder(query_input_ids, query_attention_mask)
        d_embs, d_mask = self.encoder(doc_input_ids, doc_attention_mask)

        if self.cqi is not None:
            importance = self.cqi(q_embs, q_mask)
        else:
            importance = q_mask.float()

        scores = []
        for i in range(q_embs.shape[0]):
            s = fluke_score(
                q_embs[i], d_embs[i], importance[i],
                tir_module=self.tir,
                query_mask=q_mask[i], doc_mask=d_mask[i],
                topk=self.topk if self.use_soft_topk else None,
                temperature=self.temperature,
                max_query_tokens=self.query_max_length,
            )
            scores.append(s)
        return torch.stack(scores)
