"""BEIR benchmark evaluation for late interaction retrieval models.

Evaluates on standard BEIR datasets (SciFact, NFCorpus, FiQA, etc.)
using standard IR metrics: nDCG@10, MAP@10, Recall@100.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from ..indexing.indexer import TokenEmbeddingIndex
from ..indexing.searcher import LatentSearcher


def load_beir_dataset(dataset_name: str, data_dir: str = "datasets"):
    """Load a BEIR dataset from HuggingFace or local cache.

    Returns:
        corpus: dict[doc_id] -> {"title": str, "text": str}
        queries: dict[query_id] -> str
        qrels: dict[query_id] -> dict[doc_id] -> relevance_score
    """
    from datasets import load_dataset

    cache_path = Path(data_dir) / dataset_name
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load from HuggingFace BEIR datasets
    # BEIR datasets are available as BeIR/<dataset_name>
    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    print(f"Loading BEIR dataset: {dataset_name}")

    if dataset_name == "scifact":
        ds = load_dataset("BeIR/scifact", trust_remote_code=True)
        corpus_ds = load_dataset("BeIR/scifact-corpus", trust_remote_code=True)

        for row in corpus_ds["corpus"]:
            corpus[str(row["_id"])] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
        for row in ds["queries"]:
            queries[str(row["_id"])] = row["text"]
        for row in ds["validation"]:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    elif dataset_name == "nfcorpus":
        ds = load_dataset("BeIR/nfcorpus", trust_remote_code=True)
        corpus_ds = load_dataset("BeIR/nfcorpus-corpus", trust_remote_code=True)

        for row in corpus_ds["corpus"]:
            corpus[str(row["_id"])] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
        for row in ds["queries"]:
            queries[str(row["_id"])] = row["text"]
        # nfcorpus has test split
        for row in ds["test"]:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    elif dataset_name == "fiqa":
        ds = load_dataset("BeIR/fiqa", trust_remote_code=True)
        corpus_ds = load_dataset("BeIR/fiqa-corpus", trust_remote_code=True)

        for row in corpus_ds["corpus"]:
            corpus[str(row["_id"])] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
        for row in ds["queries"]:
            queries[str(row["_id"])] = row["text"]
        for row in ds["test"]:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    elif dataset_name == "scidocs":
        ds = load_dataset("BeIR/scidocs", trust_remote_code=True)
        corpus_ds = load_dataset("BeIR/scidocs-corpus", trust_remote_code=True)

        for row in corpus_ds["corpus"]:
            corpus[str(row["_id"])] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
        for row in ds["queries"]:
            queries[str(row["_id"])] = row["text"]
        for row in ds["test"]:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Filter queries to only those with qrels
    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries with relevance judgments")
    print(f"  Qrels: {sum(len(v) for v in qrels.values())} total judgments")

    return corpus, queries, dict(qrels)


def compute_metrics(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = [10, 100],
) -> dict[str, float]:
    """Compute standard IR metrics: nDCG, MAP, Recall.

    Args:
        results: query_id -> {doc_id: score}
        qrels: query_id -> {doc_id: relevance}
        k_values: cutoff values for metrics

    Returns:
        Dict of metric_name -> value
    """
    import pytrec_eval

    metrics = {}

    for k in k_values:
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {f"ndcg_cut_{k}", f"map_cut_{k}", f"recall_{k}"}
        )
        scores = evaluator.evaluate(results)

        ndcg_scores = [s[f"ndcg_cut_{k}"] for s in scores.values()]
        map_scores = [s[f"map_cut_{k}"] for s in scores.values()]
        recall_scores = [s[f"recall_{k}"] for s in scores.values()]

        metrics[f"nDCG@{k}"] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
        metrics[f"MAP@{k}"] = sum(map_scores) / len(map_scores) if map_scores else 0
        metrics[f"Recall@{k}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    return metrics


class BEIREvaluator:
    """End-to-end evaluator for late interaction models on BEIR benchmarks."""

    def __init__(self, dataset_name: str, data_dir: str = "datasets"):
        self.dataset_name = dataset_name
        self.corpus, self.queries, self.qrels = load_beir_dataset(
            dataset_name, data_dir
        )

    def get_corpus_texts(self) -> tuple[list[str], list[str]]:
        """Get corpus document IDs and texts."""
        doc_ids = list(self.corpus.keys())
        texts = []
        for doc_id in doc_ids:
            doc = self.corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                texts.append(f"{title} {text}")
            else:
                texts.append(text)
        return doc_ids, texts

    def evaluate_model(
        self,
        model,
        model_type: str = "colbert",
        batch_size: int = 32,
        top_k: int = 100,
    ) -> dict[str, float]:
        """Full evaluation pipeline for a model on this dataset.

        Args:
            model: ColBERTModel or FLUKEModel
            model_type: "colbert" or "fluke"
            batch_size: encoding batch size
            top_k: retrieval depth

        Returns:
            Dict of metric values.
        """
        model.eval()

        # Step 1: Encode corpus
        doc_ids, doc_texts = self.get_corpus_texts()
        print(f"Encoding {len(doc_texts)} documents...")
        doc_embs = model.encode_documents(doc_texts, batch_size=batch_size, show_progress=True)

        # Step 2: Build index
        index = TokenEmbeddingIndex()
        index.add_batch(doc_ids, doc_embs)
        print(f"Index size: {index.storage_size_mb():.1f} MB ({index.num_docs} docs)")

        # Step 3: Encode queries
        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        print(f"Encoding {len(query_texts)} queries...")

        if model_type == "fluke":
            query_data = model.encode_queries(query_texts, batch_size=batch_size)
        else:
            query_data = model.encode_queries(query_texts, batch_size=batch_size)

        # Step 4: Search
        tir_module = getattr(model, "tir", None)
        fluke_plus_ref = model if model_type == "fluke_plus" else None
        searcher = LatentSearcher(
            index,
            scoring=model_type,
            tir_module=tir_module,
            max_query_tokens=getattr(model, "query_max_length", 32),
            fluke_plus_model=fluke_plus_ref,
        )

        print(f"Searching {len(query_ids)} queries...")
        search_results = searcher.batch_search(
            query_data, top_k=top_k, show_progress=True
        )

        # Step 5: Format results for evaluation
        formatted_results = {}
        for q_idx, results_list in search_results.items():
            qid = query_ids[q_idx]
            formatted_results[qid] = {
                doc_id: score for doc_id, score in results_list
            }

        # Step 6: Compute metrics
        metrics = compute_metrics(formatted_results, self.qrels)
        return metrics
