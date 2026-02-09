"""Token embedding index for late interaction retrieval."""

import numpy as np
import torch


class TokenEmbeddingIndex:
    """Stores pre-computed per-token document embeddings for retrieval.

    Each document is stored as a variable-length set of token embeddings.
    Supports brute-force search (exact) for evaluation purposes.
    """

    def __init__(self):
        self.doc_embeddings: list[torch.Tensor] = []
        self.doc_ids: list[str] = []
        self.num_docs = 0

    def add(self, doc_id: str, embeddings: torch.Tensor):
        """Add a document's token embeddings to the index.

        Args:
            doc_id: unique document identifier
            embeddings: (num_tokens, dim) tensor of token embeddings
        """
        self.doc_embeddings.append(embeddings)
        self.doc_ids.append(doc_id)
        self.num_docs += 1

    def add_batch(
        self, doc_ids: list[str], embeddings_list: list[tuple[torch.Tensor, torch.Tensor]]
    ):
        """Add multiple documents at once.

        Args:
            doc_ids: list of document identifiers
            embeddings_list: list of (embeddings, mask) tuples
        """
        for doc_id, (embs, _mask) in zip(doc_ids, embeddings_list):
            self.add(doc_id, embs)

    def get(self, idx: int) -> tuple[str, torch.Tensor]:
        """Get document by index."""
        return self.doc_ids[idx], self.doc_embeddings[idx]

    def get_all_embeddings(self) -> list[torch.Tensor]:
        """Get all document embeddings."""
        return self.doc_embeddings

    def save(self, path: str):
        """Save index to disk."""
        torch.save(
            {"doc_ids": self.doc_ids, "doc_embeddings": self.doc_embeddings},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TokenEmbeddingIndex":
        """Load index from disk."""
        data = torch.load(path, weights_only=False)
        index = cls()
        index.doc_ids = data["doc_ids"]
        index.doc_embeddings = data["doc_embeddings"]
        index.num_docs = len(index.doc_ids)
        return index

    def storage_size_mb(self) -> float:
        """Estimate storage size in MB."""
        total_bytes = sum(
            e.numel() * e.element_size() for e in self.doc_embeddings
        )
        return total_bytes / (1024 * 1024)
