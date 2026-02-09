"""FLUKE+: Enhanced FLUKE with Multi-Granularity Scoring and Adaptive Calibration.

FLUKE+ extends FLUKE with two additional innovations:

4. **Multi-Granularity Scoring (MGS)**: Match query and document at unigram,
   bigram, and trigram levels via learned 1D convolutions. This captures
   multi-word semantic units that single-token MaxSim misses.

5. **Adaptive Score Calibration (ASC)**: Calibrate per-token scores based
   on their distribution statistics (max, mean, std, peak-above-mean).
   This dynamically distinguishes discriminative from non-discriminative
   matches, going beyond CQI's static importance weighting.

Full FLUKE+ scoring pipeline:
    1. Encode query/doc into per-token embeddings (shared with ColBERTv2)
    2. CQI: compute per-token importance weights from query context
    3. ASC: calibrate per-token scores by distribution statistics
    4. MGS: match at multiple n-gram granularities
    5. TIR: capture cross-term dependencies via MLP residual
    6. Combine: weighted sum of all components

All document representations remain identical to ColBERTv2 (per-token
embeddings + n-gram embeddings computed offline), preserving indexing efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TokenEncoder
from .fluke_model import ContextualQueryImportance
from ..scoring.fluke_scoring import TokenInteractionResidual
from ..scoring.multi_granularity import MultiGranularityScorer
from ..scoring.adaptive_calibration import AdaptiveScoreCalibrator, calibrated_score


class FLUKEPlusModel(nn.Module):
    """FLUKE+: All FLUKE innovations plus MGS and ASC.

    Components (all optional, can be ablated):
        1. CQI: Contextual Query Importance (from FLUKE)
        2. SoftTopK: Soft Top-K aggregation (from FLUKE)
        3. TIR: Token Interaction Residual (from FLUKE)
        4. MGS: Multi-Granularity Scoring (new)
        5. ASC: Adaptive Score Calibration (new)
    """

    def __init__(
        self,
        model_name: str = "small",
        embedding_dim: int = 128,
        query_max_length: int = 32,
        doc_max_length: int = 180,
        topk: int = 3,
        temperature: float = 0.1,
        # FLUKE components
        use_cqi: bool = True,
        use_soft_topk: bool = True,
        use_tir: bool = True,
        # New FLUKE+ components
        use_mgs: bool = True,
        use_asc: bool = True,
        max_ngram: int = 3,
    ):
        super().__init__()
        self.encoder = TokenEncoder(model_name, embedding_dim)
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.embedding_dim = embedding_dim
        self.topk = topk
        self.temperature = temperature
        self.use_cqi = use_cqi
        self.use_soft_topk = use_soft_topk
        self.use_tir = use_tir
        self.use_mgs = use_mgs
        self.use_asc = use_asc

        # CQI: Contextual Query Importance
        if use_cqi:
            self.cqi = ContextualQueryImportance(embedding_dim)
        else:
            self.cqi = None

        # TIR: Token Interaction Residual
        if use_tir:
            self.tir = TokenInteractionResidual(
                max_query_tokens=query_max_length, hidden_dim=64
            )
        else:
            self.tir = None

        # MGS: Multi-Granularity Scoring
        if use_mgs:
            self.mgs = MultiGranularityScorer(
                embedding_dim=embedding_dim, max_kernel=max_ngram
            )
        else:
            self.mgs = None

        # ASC: Adaptive Score Calibration
        if use_asc:
            self.asc = AdaptiveScoreCalibrator(hidden_dim=32)
        else:
            self.asc = None

        # Learned weight for combining ASC with base scoring
        # When both ASC and base scoring are active, learn how to blend them
        if use_asc:
            self.asc_blend = nn.Parameter(torch.tensor(0.5))

    def _compute_base_score(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
        importance_weights: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute base FLUKE score (CQI + optional SoftTopK).

        Returns:
            base_score: scalar score
            per_token_scores: (nq,) per-token MaxSim scores
        """
        from ..scoring.fluke_scoring import importance_weighted_maxsim

        topk = self.topk if self.use_soft_topk else None
        weighted_scores, per_token_scores = importance_weighted_maxsim(
            query_embs, doc_embs, importance_weights,
            query_mask, doc_mask, topk=topk, temperature=self.temperature,
        )
        return weighted_scores.sum(), per_token_scores

    def score(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
        importance_weights: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a single query-document pair using full FLUKE+ scoring."""
        total_score = torch.tensor(0.0, device=query_embs.device)

        # Base FLUKE score (CQI + SoftTopK weighted MaxSim)
        base_score, per_token_scores = self._compute_base_score(
            query_embs, doc_embs, importance_weights, query_mask, doc_mask
        )

        # ASC: Adaptive Score Calibration
        if self.asc is not None:
            asc_score, _ = calibrated_score(
                query_embs, doc_embs, self.asc,
                importance_weights=importance_weights,
                query_mask=query_mask, doc_mask=doc_mask,
            )
            # Blend base and ASC scores
            blend = torch.sigmoid(self.asc_blend)
            total_score = blend * base_score + (1 - blend) * asc_score
        else:
            total_score = base_score

        # MGS: Multi-Granularity Scoring (additive contribution)
        if self.mgs is not None:
            mgs_score, _ = self.mgs(query_embs, doc_embs, query_mask, doc_mask)
            total_score = total_score + mgs_score

        # TIR: Token Interaction Residual
        if self.tir is not None:
            nq = per_token_scores.shape[0]
            padded = torch.zeros(
                self.query_max_length, device=per_token_scores.device
            )
            padded[:nq] = per_token_scores
            if query_mask is not None:
                mask_padded = torch.zeros(
                    self.query_max_length, device=per_token_scores.device
                )
                mask_padded[:nq] = query_mask.float()
                padded = padded * mask_padded
            residual = self.tir(padded.unsqueeze(0))
            total_score = total_score + residual.squeeze()

        return total_score

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
            scores.append(
                self.score(query_embs, doc_embs, importance_weights, query_mask)
            )
        return torch.stack(scores)

    def encode_queries(
        self, queries: list[str], batch_size: int = 32
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode queries into per-token embeddings with importance weights."""
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

        Document encoding is identical to ColBERTv2/FLUKE — all innovations
        operate at query time or use pre-computable n-gram embeddings.
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
            s = self.score(
                q_embs[i], d_embs[i], importance[i],
                query_mask=q_mask[i], doc_mask=d_mask[i],
            )
            scores.append(s)
        return torch.stack(scores)
