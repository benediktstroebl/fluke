"""ColBERTv2 MaxSim scoring: the standard late interaction scoring function."""

import torch
import torch.nn.functional as F


def maxsim(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MaxSim score between a single query and document.

    For each query token, find its maximum similarity to any document token,
    then sum across query tokens.

    Args:
        query_embeddings: (num_query_tokens, dim)
        doc_embeddings: (num_doc_tokens, dim)
        query_mask: (num_query_tokens,) boolean mask for valid query tokens
        doc_mask: (num_doc_tokens,) boolean mask for valid doc tokens

    Returns:
        Scalar relevance score.
    """
    # (num_query_tokens, num_doc_tokens)
    sim_matrix = query_embeddings @ doc_embeddings.T

    if doc_mask is not None:
        sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))

    # MaxSim: for each query token, take max over doc tokens
    max_sims = sim_matrix.max(dim=-1).values  # (num_query_tokens,)

    if query_mask is not None:
        max_sims = max_sims * query_mask.float()

    return max_sims.sum()


def batched_maxsim(
    query_embeddings: torch.Tensor,
    doc_embeddings_list: list[torch.Tensor],
    query_mask: torch.Tensor | None = None,
    doc_masks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute MaxSim scores between one query and multiple documents.

    Args:
        query_embeddings: (num_query_tokens, dim)
        doc_embeddings_list: list of (num_doc_tokens_i, dim) tensors
        query_mask: (num_query_tokens,) boolean mask
        doc_masks: list of (num_doc_tokens_i,) boolean masks

    Returns:
        (num_docs,) tensor of relevance scores.
    """
    scores = []
    for i, doc_embs in enumerate(doc_embeddings_list):
        doc_mask = doc_masks[i] if doc_masks is not None else None
        scores.append(maxsim(query_embeddings, doc_embs, query_mask, doc_mask))
    return torch.stack(scores)


def padded_maxsim(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MaxSim for batched, padded tensors.

    Args:
        query_embeddings: (batch_q, num_query_tokens, dim)
        doc_embeddings: (batch_d, num_doc_tokens, dim)
        query_mask: (batch_q, num_query_tokens)
        doc_mask: (batch_d, num_doc_tokens)

    Returns:
        (batch_q, batch_d) matrix of relevance scores.
    """
    # (batch_q, num_query_tokens, dim) x (batch_d, num_doc_tokens, dim)^T
    # -> (batch_q, batch_d, num_query_tokens, num_doc_tokens)
    sim = torch.einsum("qid,djd->qjid", query_embeddings, doc_embeddings)
    # Note: this einsum does q_tokens x d_tokens for each (q, d) pair
    # Reshape: (batch_q, num_query_tokens, dim) @ (batch_d, dim, num_doc_tokens)
    # -> use bmm-like approach

    batch_q, nq, dim = query_embeddings.shape
    batch_d, nd, _ = doc_embeddings.shape

    # (batch_q, nq, dim) -> (batch_q, 1, nq, dim)
    q = query_embeddings.unsqueeze(1)
    # (batch_d, nd, dim) -> (1, batch_d, dim, nd)
    d = doc_embeddings.permute(0, 2, 1).unsqueeze(0)

    # (batch_q, batch_d, nq, nd)
    sim = torch.matmul(q, d)

    if doc_mask is not None:
        # (1, batch_d, 1, nd)
        sim = sim.masked_fill(~doc_mask.unsqueeze(0).unsqueeze(2), float("-inf"))

    # Max over doc tokens: (batch_q, batch_d, nq)
    max_sims = sim.max(dim=-1).values

    if query_mask is not None:
        # (batch_q, 1, nq)
        max_sims = max_sims * query_mask.unsqueeze(1).float()

    # Sum over query tokens: (batch_q, batch_d)
    return max_sims.sum(dim=-1)
