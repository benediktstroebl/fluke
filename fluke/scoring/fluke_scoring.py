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
    compute_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """MaxSim with per-query-token importance weighting.

    Args:
        query_embeddings: (num_query_tokens, dim)
        doc_embeddings: (num_doc_tokens, dim)
        importance_weights: (num_query_tokens,) learned importance per query token
        query_mask: (num_query_tokens,) boolean mask
        doc_mask: (num_doc_tokens,) boolean mask
        topk: if set, use soft top-K instead of hard max
        temperature: softmax temperature for soft top-K
        compute_stats: if True, compute per-token mean/std similarity stats

    Returns:
        weighted_scores, per_token_scores, per_token_mean (or None), per_token_std (or None)
    """
    per_token_mean = None
    per_token_std = None

    if topk is not None:
        # Vectorized soft top-K aggregation
        sim_matrix = query_embeddings @ doc_embeddings.T  # (nq, nd)
        if doc_mask is not None:
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))
        actual_k = min(topk, sim_matrix.shape[1])
        topk_sims, _ = sim_matrix.topk(actual_k, dim=-1)  # (nq, k)
        weights = F.softmax(topk_sims / temperature, dim=-1)  # (nq, k)
        per_token_scores = (weights * topk_sims).sum(dim=-1)  # (nq,)
    else:
        # Standard hard MaxSim
        sim_matrix = query_embeddings @ doc_embeddings.T
        if doc_mask is not None:
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))
        per_token_scores = sim_matrix.max(dim=-1).values

    # Optionally compute per-token distribution stats (for enhanced TIR)
    if compute_stats:
        with torch.no_grad():
            sim_matrix_full = query_embeddings @ doc_embeddings.T
            if doc_mask is not None:
                valid_sims = sim_matrix_full.masked_fill(~doc_mask.unsqueeze(0), 0.0)
                n_valid = doc_mask.float().sum().clamp(min=1)
                per_token_mean = valid_sims.sum(dim=-1) / n_valid
                per_token_var = ((valid_sims - per_token_mean.unsqueeze(-1)) ** 2 * doc_mask.float().unsqueeze(0)).sum(dim=-1) / n_valid
            else:
                per_token_mean = sim_matrix_full.mean(dim=-1)
                per_token_var = sim_matrix_full.var(dim=-1, correction=0)
            per_token_std = per_token_var.sqrt()

    # Apply importance weights
    weighted_scores = per_token_scores * importance_weights

    if query_mask is not None:
        weighted_scores = weighted_scores * query_mask.float()

    return weighted_scores, per_token_scores, per_token_mean, per_token_std


class TokenInteractionResidual(nn.Module):
    """Lightweight MLP that captures cross-term dependencies.

    Takes the vector of per-query-token MaxSim scores (and optionally
    mean/std similarity statistics) and outputs a scalar residual correction.
    This allows the model to learn patterns like:
    - "Both terms A and B must match" (conjunction)
    - "Term A matching but B not matching is bad" (required terms)
    - "This match is discriminative (high std)" vs "stopword match (low std)"
    - Non-linear score interactions

    The MLP operates on |q|-dimensional input (typically 32 or 96 with stats),
    so it's extremely lightweight and adds negligible compute.
    """

    def __init__(self, max_query_tokens: int = 32, hidden_dim: int = 64,
                 use_score_stats: bool = False):
        super().__init__()
        self.use_score_stats = use_score_stats
        # Input: max scores + optionally mean and std per token
        input_dim = max_query_tokens * (3 if use_score_stats else 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

    def forward(self, per_token_scores: torch.Tensor,
                per_token_mean: torch.Tensor | None = None,
                per_token_std: torch.Tensor | None = None) -> torch.Tensor:
        """Compute interaction residual from per-token match scores.

        Args:
            per_token_scores: (batch, max_query_tokens) per-token MaxSim scores.
            per_token_mean: (batch, max_query_tokens) mean similarity per token.
            per_token_std: (batch, max_query_tokens) std similarity per token.

        Returns:
            (batch,) residual score corrections.
        """
        if self.use_score_stats and per_token_mean is not None and per_token_std is not None:
            x = torch.cat([per_token_scores, per_token_mean, per_token_std], dim=-1)
        else:
            x = per_token_scores
        return self.net(x).squeeze(-1)


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
    disc_scale: float = 0.0,
    disc_range: tuple[float, float] = (0.5, 1.5),
    coverage_weight: float = 0.0,
) -> torch.Tensor:
    """Complete FLUKE scoring function.

    Combines:
    1. Importance-weighted MaxSim (or soft top-K)
    2. Token Interaction Residual (with optional score distribution stats)

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
        disc_scale: sigmoid scale for discriminativeness reweighting
        disc_range: (lo, hi) bounds for discriminativeness weights
        coverage_weight: strength of coverage bonus (0 = disabled)

    Returns:
        Scalar relevance score.
    """
    need_stats = tir_module is not None and tir_module.use_score_stats
    weighted_scores, per_token_scores, per_token_mean, per_token_std = importance_weighted_maxsim(
        query_embeddings, doc_embeddings, importance_weights,
        query_mask, doc_mask, topk=topk, temperature=temperature,
        compute_stats=need_stats,
    )

    if disc_scale > 0:
        sim_matrix_disc = query_embeddings @ doc_embeddings.T
        if doc_mask is not None:
            valid_sims_disc = sim_matrix_disc.masked_fill(~doc_mask.unsqueeze(0), 0.0)
            n_valid_disc = doc_mask.float().sum().clamp(min=1)
            mean_disc = valid_sims_disc.sum(dim=-1) / n_valid_disc
        else:
            mean_disc = sim_matrix_disc.mean(dim=-1)
        max_disc = sim_matrix_disc.max(dim=-1).values if doc_mask is None else \
            sim_matrix_disc.masked_fill(~doc_mask.unsqueeze(0), float("-inf")).max(dim=-1).values
        peak_above_mean = (max_disc - mean_disc).clamp(min=0)
        pam_std = peak_above_mean.std().clamp(min=1e-6)
        effective_scale = (disc_scale * pam_std).detach()
        disc_weights = (1.0 + effective_scale * peak_above_mean).detach()
        disc_weighted = weighted_scores * disc_weights
        if query_mask is not None:
            disc_weighted = disc_weighted * query_mask.float()
    else:
        disc_weighted = weighted_scores

    cov_factor = torch.tensor(1.0, device=weighted_scores.device)
    if coverage_weight > 0:
        active = disc_weighted[query_mask] if query_mask is not None else disc_weighted
        if active.numel() > 1:
            min_s = active.min()
            mean_s = active.mean().clamp(min=1e-6)
            coverage = (min_s / mean_s).clamp(0, 1)
            cov_factor = (1.0 + coverage_weight * coverage).detach()

    base_score = disc_weighted.sum() * cov_factor

    if tir_module is not None:
        # Pad per_token_scores to fixed size for TIR
        nq = per_token_scores.shape[0]
        padded = torch.zeros(max_query_tokens, device=per_token_scores.device)
        padded[:nq] = per_token_scores

        # Pad mean and std if available and TIR uses them
        padded_mean = None
        padded_std = None
        if tir_module.use_score_stats and per_token_mean is not None:
            padded_mean = torch.zeros(max_query_tokens, device=per_token_scores.device)
            padded_mean[:nq] = per_token_mean
            padded_std = torch.zeros(max_query_tokens, device=per_token_scores.device)
            padded_std[:nq] = per_token_std

        if query_mask is not None:
            mask_padded = torch.zeros(max_query_tokens, device=per_token_scores.device)
            mask_padded[:nq] = query_mask.float()
            padded = padded * mask_padded
            if padded_mean is not None:
                padded_mean = padded_mean * mask_padded
                padded_std = padded_std * mask_padded

        residual = tir_module(
            padded.unsqueeze(0),
            padded_mean.unsqueeze(0) if padded_mean is not None else None,
            padded_std.unsqueeze(0) if padded_std is not None else None,
        )
        return base_score + residual.squeeze()

    return base_score
