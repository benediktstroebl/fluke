"""Search/retrieval over token embedding indices."""

import torch
from tqdm import tqdm

from .indexer import TokenEmbeddingIndex
from ..scoring.maxsim import maxsim
from ..scoring.fluke_scoring import fluke_score, TokenInteractionResidual


class LatentSearcher:
    """Brute-force searcher over a TokenEmbeddingIndex.

    Computes exact scores between a query and all documents in the index.
    Suitable for evaluation on small-to-medium collections (up to ~100K docs).
    """

    def __init__(
        self,
        index: TokenEmbeddingIndex,
        scoring: str = "maxsim",
        topk_param: int = 3,
        temperature: float = 0.1,
        tir_module: TokenInteractionResidual | None = None,
        max_query_tokens: int = 32,
        fluke_plus_model=None,
    ):
        self.index = index
        self.scoring = scoring
        self.topk_param = topk_param
        self.temperature = temperature
        self.tir_module = tir_module
        self.max_query_tokens = max_query_tokens
        self.fluke_plus_model = fluke_plus_model

    def search(
        self,
        query_embs: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        importance_weights: torch.Tensor | None = None,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Search the index for the top-K most relevant documents.

        Args:
            query_embs: (num_query_tokens, dim) query token embeddings
            query_mask: (num_query_tokens,) boolean mask
            importance_weights: (num_query_tokens,) for FLUKE scoring
            top_k: number of results to return

        Returns:
            List of (doc_id, score) tuples, sorted by descending score.
        """
        scores = []

        for i in range(self.index.num_docs):
            doc_embs = self.index.doc_embeddings[i]

            if self.scoring == "fluke_plus" and self.fluke_plus_model is not None:
                iw = importance_weights if importance_weights is not None else torch.ones(query_embs.shape[0])
                score = self.fluke_plus_model.score(
                    query_embs, doc_embs, iw, query_mask=query_mask,
                )
            elif self.scoring == "fluke" and importance_weights is not None:
                score = fluke_score(
                    query_embs, doc_embs, importance_weights,
                    tir_module=self.tir_module,
                    query_mask=query_mask,
                    topk=self.topk_param,
                    temperature=self.temperature,
                    max_query_tokens=self.max_query_tokens,
                )
            else:
                score = maxsim(query_embs, doc_embs, query_mask)

            scores.append(score.item())

        # Get top-K indices
        scores_tensor = torch.tensor(scores)
        topk_indices = scores_tensor.topk(min(top_k, len(scores))).indices.tolist()

        results = []
        for idx in topk_indices:
            results.append((self.index.doc_ids[idx], scores[idx]))

        return results

    def batch_search(
        self,
        query_embs_list: list[tuple],
        top_k: int = 100,
        show_progress: bool = False,
    ) -> dict[int, list[tuple[str, float]]]:
        """Search for multiple queries.

        Args:
            query_embs_list: list of tuples, either:
                - (embs, mask) for MaxSim
                - (embs, mask, weights) for FLUKE
            top_k: number of results per query
            show_progress: show tqdm progress bar

        Returns:
            Dict mapping query index to list of (doc_id, score).
        """
        results = {}
        iterator = enumerate(query_embs_list)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Searching")

        for q_idx, q_data in iterator:
            if len(q_data) == 3:
                embs, mask, weights = q_data
                results[q_idx] = self.search(embs, mask, weights, top_k)
            else:
                embs, mask = q_data
                results[q_idx] = self.search(embs, mask, top_k=top_k)

        return results
