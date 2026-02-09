"""FLUKE scoring functions: improvements over ColBERTv2's MaxSim.

Three key innovations:
1. Importance-Weighted MaxSim (IW-MaxSim): Weight query token contributions
   by their contextual importance.
2. Soft Top-K Aggregation: Instead of hard max, aggregate top-K similarities
   with softmax weighting for robustness.
3. Token Interaction Residual (TIR): A lightweight MLP that captures cross-term
   dependencies from the vector of per-token match scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_topk_sim(
    query_token: torch.Tensor,
    doc_embeddings: torch.Tensor,
    k: int = 3,
    temperature: float = 0.1,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Top-K similarity: aggregate top-K matches with softmax weighting.

    Instead of hard max (ColBERTv2), this smooths over the top-K matches,
    making scoring more robust to spurious high-similarity tokens.

    Args:
        query_token: (dim,) single query token embedding
        doc_embeddings: (num_doc_tokens, dim) document token embeddings
        k: number of top matches to aggregate
        temperature: softmax temperature (lower = sharper, closer to hard max)
        doc_mask: (num_doc_tokens,) boolean mask for valid tokens

    Returns:
        Scalar soft top-K similarity score.
    """
    sims = doc_embeddings @ query_token  # (num_doc_tokens,)

    if doc_mask is not None:
        sims = sims.masked_fill(~doc_mask, float("-inf"))

    actual_k = min(k, sims.shape[0])
    topk_sims, _ = sims.topk(actual_k)  # (k,)

    # Softmax-weighted aggregation of top-K similarities
    weights = F.softmax(topk_sims / temperature, dim=0)
    return (weights * topk_sims).sum()


def importance_weighted_maxsim(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    importance_weights: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
    topk: int | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """MaxSim with per-query-token importance weighting.

    Args:
        query_embeddings: (num_query_tokens, dim)
        doc_embeddings: (num_doc_tokens, dim)
        importance_weights: (num_query_tokens,) learned importance per query token
        query_mask: (num_query_tokens,) boolean mask
        doc_mask: (num_doc_tokens,) boolean mask
        topk: if set, use soft top-K instead of hard max
        temperature: softmax temperature for soft top-K

    Returns:
        Scalar relevance score.
    """
    if topk is not None:
        # Use soft top-K aggregation
        scores = []
        for i in range(query_embeddings.shape[0]):
            s = soft_topk_sim(
                query_embeddings[i], doc_embeddings, k=topk,
                temperature=temperature, doc_mask=doc_mask,
            )
            scores.append(s)
        per_token_scores = torch.stack(scores)
    else:
        # Standard hard MaxSim
        sim_matrix = query_embeddings @ doc_embeddings.T
        if doc_mask is not None:
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))
        per_token_scores = sim_matrix.max(dim=-1).values

    # Apply importance weights
    weighted_scores = per_token_scores * importance_weights

    if query_mask is not None:
        weighted_scores = weighted_scores * query_mask.float()

    return weighted_scores, per_token_scores


class TokenInteractionResidual(nn.Module):
    """Lightweight MLP that captures cross-term dependencies.

    Takes the vector of per-query-token MaxSim scores and outputs a scalar
    residual correction. This allows the model to learn patterns like:
    - "Both terms A and B must match" (conjunction)
    - "Term A matching but B not matching is bad" (required terms)
    - Non-linear score interactions

    The MLP operates on |q|-dimensional input (typically 32), so it's
    extremely lightweight and adds negligible compute.
    """

    def __init__(self, max_query_tokens: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_query_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to near-zero so TIR starts as a small correction
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, per_token_scores: torch.Tensor) -> torch.Tensor:
        """Compute interaction residual from per-token match scores.

        Args:
            per_token_scores: (batch, max_query_tokens) per-token MaxSim scores,
                padded to max_query_tokens.

        Returns:
            (batch,) residual score corrections.
        """
        return self.net(per_token_scores).squeeze(-1)


def fluke_score(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    importance_weights: torch.Tensor,
    tir_module: TokenInteractionResidual | None = None,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
    topk: int = 3,
    temperature: float = 0.1,
    max_query_tokens: int = 32,
) -> torch.Tensor:
    """Complete FLUKE scoring function.

    Combines:
    1. Importance-weighted MaxSim (or soft top-K)
    2. Token Interaction Residual

    Args:
        query_embeddings: (num_query_tokens, dim)
        doc_embeddings: (num_doc_tokens, dim)
        importance_weights: (num_query_tokens,)
        tir_module: optional TokenInteractionResidual module
        query_mask: (num_query_tokens,)
        doc_mask: (num_doc_tokens,)
        topk: K for soft top-K aggregation
        temperature: softmax temperature
        max_query_tokens: padding size for TIR input

    Returns:
        Scalar relevance score.
    """
    weighted_scores, per_token_scores = importance_weighted_maxsim(
        query_embeddings, doc_embeddings, importance_weights,
        query_mask, doc_mask, topk=topk, temperature=temperature,
    )

    base_score = weighted_scores.sum()

    if tir_module is not None:
        # Pad per_token_scores to fixed size for TIR
        nq = per_token_scores.shape[0]
        padded = torch.zeros(max_query_tokens, device=per_token_scores.device)
        padded[:nq] = per_token_scores
        if query_mask is not None:
            mask_padded = torch.zeros(max_query_tokens, device=per_token_scores.device)
            mask_padded[:nq] = query_mask.float()
            padded = padded * mask_padded
        residual = tir_module(padded.unsqueeze(0))
        return base_score + residual.squeeze()

    return base_score
