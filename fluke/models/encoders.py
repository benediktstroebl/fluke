"""Shared encoder components for late interaction models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenizer:
    """Hash-based word-piece tokenizer for environments without HuggingFace access.

    Uses hash-based tokenization into a fixed vocabulary. Sufficient for
    demonstrating the scoring innovations of FLUKE vs ColBERTv2 when
    trained from scratch.
    """

    def __init__(self, vocab_size: int = 5000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3

    def __call__(
        self, texts: list[str], padding: bool = True, truncation: bool = True,
        max_length: int | None = None, return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        max_len = max_length or self.max_length
        all_ids = []
        all_masks = []

        for text in texts:
            tokens = [self.cls_token_id]
            for word in text.lower().split():
                # 3-character subword pieces via hashing
                for i in range(0, len(word), 3):
                    chunk = word[i:i+3]
                    token_id = (hash(chunk) % (self.vocab_size - 4)) + 4
                    tokens.append(token_id)
            tokens.append(self.sep_token_id)

            if truncation and len(tokens) > max_len:
                tokens = tokens[:max_len - 1] + [self.sep_token_id]

            mask = [1] * len(tokens)

            if padding:
                pad_len = max_len - len(tokens)
                if pad_len > 0:
                    tokens.extend([self.pad_token_id] * pad_len)
                    mask.extend([0] * pad_len)

            all_ids.append(tokens)
            all_masks.append(mask)

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return (hash(token) % (self.vocab_size - 4)) + 4


class SmallTransformerEncoder(nn.Module):
    """Lightweight transformer encoder for training from scratch.

    A small but functional transformer that can learn meaningful representations
    when trained on sufficient data.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        intermediate_size: int = 512,
        max_position: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size = hidden_size

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        src_key_padding_mask = ~attention_mask.bool()
        hidden = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        return hidden


class TokenEncoder(nn.Module):
    """Token-level encoder for late interaction retrieval.

    Encodes text into per-token embeddings (not a single vector).
    Used as the backbone for both ColBERT and FLUKE models.

    Supports two modes:
    1. Pre-trained HuggingFace model (model_name = 'distilbert-base-uncased' etc.)
    2. Small from-scratch transformer (model_name = 'small')
    """

    def __init__(
        self,
        model_name: str = "small",
        embedding_dim: int = 128,
        max_length: int = 512,
        vocab_size: int = 5000,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        if model_name == "small":
            self.transformer = SmallTransformerEncoder(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
            self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=max_length)
            actual_hidden = hidden_size
        else:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.transformer = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                actual_hidden = self.transformer.config.hidden_size
            except Exception:
                print(f"Could not load {model_name}, falling back to small transformer")
                self.transformer = SmallTransformerEncoder(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
                self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=max_length)
                actual_hidden = hidden_size

        self.linear = nn.Linear(actual_hidden, embedding_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input tokens into per-token embeddings.

        Returns:
            token_embeddings: (batch, seq_len, embedding_dim) L2-normalized
            mask: (batch, seq_len) boolean attention mask
        """
        if isinstance(self.transformer, SmallTransformerEncoder):
            hidden = self.transformer(input_ids, attention_mask)
        else:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

        embeddings = self.linear(hidden)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, attention_mask.bool()

    def tokenize(
        self, texts: list[str], max_length: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Tokenize a list of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
        )
