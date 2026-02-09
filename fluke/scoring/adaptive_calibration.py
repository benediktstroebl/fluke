"""Adaptive Score Calibration (ASC): normalize per-token scores before aggregation.

Problem: In ColBERTv2's MaxSim, different query tokens produce match scores
with very different distributions. A common word like "the" will have high
MaxSim scores against almost any document (many tokens match it), while a
rare technical term might have uniformly low scores. When we sum these scores,
the common words dominate, drowning out the signal from discriminative terms.

CQI (Contextual Query Importance) partially addresses this by learning static
per-token weights. But ASC goes further: it calibrates each per-token score
based on the actual score distribution for that query token across the
specific document. This is a dynamic, input-dependent calibration.

For each query token i, we compute statistics of its similarity distribution
over document tokens:
    - max_sim: the best match score
    - mean_sim: average similarity across doc tokens
    - std_sim: spread of similarities

Then we apply a residual calibration:
    calibrated_score_i = max_sim_i + MLP([max_sim_i, mean_sim_i, std_sim_i, peak])

Key insight: Using a RESIDUAL architecture (max_sim + correction) means the
calibrator starts identical to ColBERTv2 and learns adjustments. The correction
term captures how "peaky" the match is — a high peak above mean indicates a
specific, discriminative match that should be upweighted relative to a
non-discriminative match.

This calibration operates on per-query-token statistics computed from the
similarity matrix, which is already computed in MaxSim. The overhead is
minimal: just one MLP forward pass per query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveScoreCalibrator(nn.Module):
    """Calibrates per-token MaxSim scores based on score distribution statistics.

    Uses a RESIDUAL design: output = max_sim + small_correction.
    This ensures the model starts identical to ColBERTv2 (zero-init correction)
    and gradually learns to adjust scores based on discriminativeness.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Input: [max_sim, mean_sim, std_sim, peak_above_mean]
        # Output: scalar correction to max_sim
        self.correction_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-initialize the correction so we start exactly at ColBERTv2
        self._init_zero_correction()

    def _init_zero_correction(self):
        """Initialize correction network to output ~0 (start as ColBERTv2)."""
        for layer in self.correction_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)
        # Final layer outputs near-zero
        with torch.no_grad():
            self.correction_net[-1].weight.data.fill_(0.0)
            self.correction_net[-1].bias.data.fill_(0.0)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute calibrated per-token scores.

        Args:
            query_embeddings: (num_query_tokens, dim)
            doc_embeddings: (num_doc_tokens, dim)
            query_mask: (num_query_tokens,)
            doc_mask: (num_doc_tokens,)

        Returns:
            calibrated_scores: (num_query_tokens,) calibrated per-token scores
            raw_max_sims: (num_query_tokens,) original MaxSim scores (for TIR)
        """
        # Compute full similarity matrix
        sim_matrix = query_embeddings @ doc_embeddings.T  # (nq, nd)

        if doc_mask is not None:
            sim_matrix_masked = sim_matrix.masked_fill(
                ~doc_mask.unsqueeze(0), float("-inf")
            )
            sim_for_stats = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), 0.0)
            n_valid_docs = doc_mask.float().sum().clamp(min=1)
        else:
            sim_matrix_masked = sim_matrix
            sim_for_stats = sim_matrix
            n_valid_docs = float(doc_embeddings.shape[0])

        # Per-query-token statistics
        max_sims = sim_matrix_masked.max(dim=-1).values  # (nq,)
        mean_sims = sim_for_stats.sum(dim=-1) / n_valid_docs  # (nq,)
        sq_diff = (sim_for_stats - mean_sims.unsqueeze(-1)) ** 2
        if doc_mask is not None:
            sq_diff = sq_diff * doc_mask.float().unsqueeze(0)
        std_sims = (sq_diff.sum(dim=-1) / n_valid_docs).sqrt()  # (nq,)

        peak_above_mean = max_sims - mean_sims  # (nq,)

        # Stack features: (nq, 4)
        features = torch.stack(
            [max_sims, mean_sims, std_sims, peak_above_mean], dim=-1
        )

        # Residual calibration: max_sim + learned correction
        correction = self.correction_net(features).squeeze(-1)  # (nq,)
        calibrated = max_sims + correction

        if query_mask is not None:
            calibrated = calibrated * query_mask.float()
            max_sims = max_sims * query_mask.float()

        return calibrated, max_sims


def calibrated_score(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    calibrator: AdaptiveScoreCalibrator,
    importance_weights: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute calibrated relevance score with optional importance weighting.

    Args:
        query_embeddings: (nq, dim)
        doc_embeddings: (nd, dim)
        calibrator: AdaptiveScoreCalibrator module
        importance_weights: (nq,) optional CQI weights
        query_mask: (nq,) boolean mask
        doc_mask: (nd,) boolean mask

    Returns:
        total_score: scalar relevance score
        per_token_scores: (nq,) raw max-sim scores for TIR
    """
    calibrated_scores, raw_max_sims = calibrator(
        query_embeddings, doc_embeddings, query_mask, doc_mask
    )

    if importance_weights is not None:
        calibrated_scores = calibrated_scores * importance_weights

    total = calibrated_scores.sum()
    return total, raw_max_sims
