"""Multi-Granularity Scoring (MGS): match at token, bigram, and trigram levels.

ColBERTv2's MaxSim operates purely at the individual token level: each query
token finds its best-matching document token. This misses multi-word semantic
units entirely. For example, "machine learning" as two separate tokens may
match "machine" (laundry machine) and "learning" (child learning) in an
irrelevant document, scoring higher than a relevant document about "ML".

MGS addresses this by creating n-gram-level embeddings via 1D convolutions
over token embeddings, then matching at multiple granularities. The key insight
is that 1D convolutions over L2-normalized token embeddings produce embeddings
that capture n-gram semantics, and these can be pre-computed for documents
just like single-token embeddings.

Architecture:
    Token embeddings -> [Conv1D_k2, Conv1D_k3] -> n-gram embeddings
    Score = α₁·MaxSim_unigram + α₂·MaxSim_bigram + α₃·MaxSim_trigram

The convolution weights and scale weights α are learned. Document n-gram
embeddings are computed offline (same as regular token embeddings), preserving
ColBERTv2's indexing efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramEmbedder(nn.Module):
    """Creates n-gram embeddings from token embeddings using 1D convolutions.

    For kernel size k, produces (seq_len - k + 1) n-gram embeddings, each
    capturing the semantics of k consecutive tokens.
    """

    def __init__(self, embedding_dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            padding=0,  # no padding -> output length = input - kernel + 1
            bias=True,
        )
        self.kernel_size = kernel_size
        # Initialize close to averaging (so initial n-gram ≈ mean of tokens)
        nn.init.constant_(self.conv.weight, 1.0 / (kernel_size * embedding_dim))
        nn.init.zeros_(self.conv.bias)

    def forward(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create n-gram embeddings.

        Args:
            token_embeddings: (num_tokens, dim) or (batch, num_tokens, dim)
            mask: (num_tokens,) or (batch, num_tokens)

        Returns:
            ngram_embeddings: L2-normalized n-gram embeddings
            ngram_mask: validity mask for n-grams
        """
        single = token_embeddings.dim() == 2
        if single:
            token_embeddings = token_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch, seq_len, dim = token_embeddings.shape

        # If sequence is shorter than kernel, return empty tensors
        if seq_len < self.kernel_size:
            empty_ngrams = torch.zeros(batch, 0, dim, device=token_embeddings.device)
            empty_mask = torch.zeros(batch, 0, dtype=torch.bool, device=token_embeddings.device)
            if single:
                empty_ngrams = empty_ngrams.squeeze(0)
                empty_mask = empty_mask.squeeze(0)
            return empty_ngrams, empty_mask

        # Conv1d expects (batch, channels, length)
        x = token_embeddings.permute(0, 2, 1)  # (batch, dim, seq_len)
        ngrams = self.conv(x)  # (batch, dim, seq_len - k + 1)
        ngrams = ngrams.permute(0, 2, 1)  # (batch, new_len, dim)

        # L2 normalize
        ngrams = F.normalize(ngrams, p=2, dim=-1)

        # Create mask for n-grams: an n-gram is valid only if ALL its
        # constituent tokens are valid
        if mask is not None:
            mask_float = mask.float()
            # Use avg pooling on mask to check if all tokens in window are valid
            mask_1d = mask_float.unsqueeze(1)  # (batch, 1, seq_len)
            ngram_mask_scores = F.avg_pool1d(
                mask_1d, kernel_size=self.kernel_size, stride=1, padding=0
            ).squeeze(1)  # (batch, new_len)
            # All tokens in window must be valid (avg = 1.0)
            ngram_mask = ngram_mask_scores >= (1.0 - 1e-6)
        else:
            ngram_mask = torch.ones(
                batch, ngrams.shape[1], dtype=torch.bool, device=ngrams.device
            )

        if single:
            ngrams = ngrams.squeeze(0)
            ngram_mask = ngram_mask.squeeze(0)

        return ngrams, ngram_mask


class MultiGranularityScorer(nn.Module):
    """Scores query-document pairs at multiple granularities.

    Combines unigram (standard MaxSim), bigram, and trigram matching scores
    with learned scale weights.
    """

    def __init__(self, embedding_dim: int = 128, max_kernel: int = 3):
        super().__init__()
        self.bigram_embedder = NGramEmbedder(embedding_dim, kernel_size=2)
        self.trigram_embedder = NGramEmbedder(embedding_dim, kernel_size=3) if max_kernel >= 3 else None

        # Learned scale weights for combining granularities
        # Initialize to favor unigram (ColBERTv2 behavior) with small n-gram contribution
        n_scales = 2 if max_kernel < 3 else 3
        self.scale_logits = nn.Parameter(torch.tensor(
            [2.0, 0.5, 0.3][:n_scales]  # softmax -> ~[0.7, 0.15, 0.15]
        ))
        self.max_kernel = max_kernel

    def get_scale_weights(self) -> torch.Tensor:
        """Get normalized scale weights via softmax."""
        return F.softmax(self.scale_logits, dim=0)

    def compute_ngram_maxsim(
        self,
        query_ngrams: torch.Tensor,
        doc_ngrams: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MaxSim between query n-grams and document n-grams."""
        # (num_query_ngrams, num_doc_ngrams)
        sim_matrix = query_ngrams @ doc_ngrams.T

        if doc_mask is not None:
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))

        # Max over doc n-grams for each query n-gram
        max_sims = sim_matrix.max(dim=-1).values

        if query_mask is not None:
            max_sims = max_sims * query_mask.float()

        return max_sims

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-granularity score.

        Args:
            query_embeddings: (num_query_tokens, dim)
            doc_embeddings: (num_doc_tokens, dim)
            query_mask: (num_query_tokens,)
            doc_mask: (num_doc_tokens,)

        Returns:
            total_score: scalar combined multi-granularity score
            per_token_unigram_scores: (num_query_tokens,) for downstream use
        """
        weights = self.get_scale_weights()

        # 1. Unigram MaxSim (standard ColBERTv2)
        sim_matrix = query_embeddings @ doc_embeddings.T
        if doc_mask is not None:
            sim_matrix = sim_matrix.masked_fill(~doc_mask.unsqueeze(0), float("-inf"))
        unigram_scores = sim_matrix.max(dim=-1).values
        if query_mask is not None:
            unigram_scores = unigram_scores * query_mask.float()
        unigram_total = unigram_scores.sum()

        # 2. Bigram MaxSim
        q_bigrams, q_bi_mask = self.bigram_embedder(query_embeddings, query_mask)
        d_bigrams, d_bi_mask = self.bigram_embedder(doc_embeddings, doc_mask)

        if q_bigrams.shape[0] > 0 and d_bigrams.shape[0] > 0:
            bigram_per_token = self.compute_ngram_maxsim(
                q_bigrams, d_bigrams, q_bi_mask, d_bi_mask
            )
            bigram_total = bigram_per_token.sum()
        else:
            bigram_total = torch.tensor(0.0, device=query_embeddings.device)

        # 3. Trigram MaxSim (if enabled)
        if self.trigram_embedder is not None:
            q_trigrams, q_tri_mask = self.trigram_embedder(query_embeddings, query_mask)
            d_trigrams, d_tri_mask = self.trigram_embedder(doc_embeddings, doc_mask)

            if q_trigrams.shape[0] > 0 and d_trigrams.shape[0] > 0:
                trigram_per_token = self.compute_ngram_maxsim(
                    q_trigrams, d_trigrams, q_tri_mask, d_tri_mask
                )
                trigram_total = trigram_per_token.sum()
            else:
                trigram_total = torch.tensor(0.0, device=query_embeddings.device)

            total = (
                weights[0] * unigram_total
                + weights[1] * bigram_total
                + weights[2] * trigram_total
            )
        else:
            total = weights[0] * unigram_total + weights[1] * bigram_total

        return total, unigram_scores
