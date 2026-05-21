#!/usr/bin/env python3
"""Evaluate FLUKE vs ColBERTv2 on real BEIR datasets.

Uses distilbert-base-uncased as the backbone encoder and trains on
MS MARCO-style triplets derived from BEIR training data.
"""

import gc
import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluke.models.colbert import ColBERTModel
from fluke.models.fluke_model import FLUKEModel
from fluke.evaluation.benchmarks import BEIREvaluator, compute_metrics
from fluke.indexing.indexer import TokenEmbeddingIndex
from fluke.indexing.searcher import LatentSearcher
from fluke.training.trainer import train_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_beir_via_package(dataset_name, split="test"):
    """Load BEIR dataset using the beir package (more reliable)."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "/tmp/beir_data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

    # Convert corpus format to match our evaluator
    corpus_dict = {}
    for doc_id, doc in corpus.items():
        corpus_dict[doc_id] = {"title": doc.get("title", ""), "text": doc.get("text", "")}

    print(f"  {dataset_name}: {len(corpus_dict)} docs, {len(queries)} queries, "
          f"{sum(len(v) for v in qrels.values())} judgments")
    return corpus_dict, queries, qrels


def make_triplets_from_qrels(corpus, queries, qrels, n_negatives=5, seed=42):
    """Generate training triplets from qrels."""
    rng = random.Random(seed)
    all_doc_ids = list(corpus.keys())
    triplets = []

    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        query_text = queries[qid]
        pos_doc_ids = [did for did, score in rels.items() if score > 0]

        if not pos_doc_ids:
            continue

        for _ in range(n_negatives):
            pos_did = rng.choice(pos_doc_ids)
            pos_doc = corpus[pos_did]
            pos_text = f"{pos_doc.get('title', '')} {pos_doc.get('text', '')}".strip()

            neg_did = rng.choice(all_doc_ids)
            while neg_did in pos_doc_ids:
                neg_did = rng.choice(all_doc_ids)
            neg_doc = corpus[neg_did]
            neg_text = f"{neg_doc.get('title', '')} {neg_doc.get('text', '')}".strip()

            triplets.append((query_text, pos_text, neg_text))

    rng.shuffle(triplets)
    return triplets


def run_real_beir_eval():
    print("=" * 70)
    print("REAL BEIR EVALUATION: FLUKE vs ColBERTv2")
    print("Using distilbert-base-uncased backbone")
    print("=" * 70)

    datasets = ["scifact", "nfcorpus"]
    backbone = "distilbert-base-uncased"
    num_epochs = 3
    lr = 2e-5
    bs = 8
    results = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load dataset via beir package
        corpus, queries, qrels = load_beir_via_package(ds_name)

        # Create training triplets from qrels
        triplets = make_triplets_from_qrels(corpus, queries, qrels, n_negatives=10)
        print(f"  Generated {len(triplets)} training triplets")
        triplets = triplets[:min(len(triplets), 2000)]

        # Prepare corpus texts
        doc_ids = list(corpus.keys())
        doc_texts = [f"{corpus[d].get('title','')} {corpus[d].get('text','')}".strip() for d in doc_ids]
        query_ids = [qid for qid in queries if qid in qrels]
        query_texts = [queries[qid] for qid in query_ids]

        def eval_model(model, model_type):
            model.eval()
            print(f"  Encoding {len(doc_texts)} documents...")
            doc_embs = model.encode_documents(doc_texts, batch_size=bs, show_progress=True)
            index = TokenEmbeddingIndex()
            index.add_batch(doc_ids, doc_embs)
            print(f"  Encoding {len(query_texts)} queries...")
            query_data = model.encode_queries(query_texts, batch_size=bs)
            tir = getattr(model, "tir", None)
            searcher = LatentSearcher(
                index, scoring=model_type, tir_module=tir,
                max_query_tokens=getattr(model, "query_max_length", 32),
            )
            print(f"  Searching...")
            search_results = searcher.batch_search(query_data, top_k=100, show_progress=True)
            formatted = {}
            for q_idx, rl in search_results.items():
                formatted[query_ids[q_idx]] = {d: s for d, s in rl}
            return compute_metrics(formatted, qrels)

        # ColBERTv2
        print(f"\n  --- ColBERTv2 ({backbone}) ---")
        set_seed(42)
        colbert = ColBERTModel(
            model_name=backbone, embedding_dim=128,
            query_max_length=32, doc_max_length=180,
        )
        print(f"  Training ({len(triplets)} triplets, {num_epochs} epochs)...")
        train_model(colbert, triplets, num_epochs=num_epochs, batch_size=bs, lr=lr)
        colbert_metrics = eval_model(colbert, "colbert")
        print(f"  ColBERTv2: nDCG@10={colbert_metrics['nDCG@10']:.4f}, "
              f"Recall@100={colbert_metrics['Recall@100']:.4f}")
        results[f"{ds_name}_colbert"] = colbert_metrics
        del colbert
        gc.collect()

        # FLUKE
        print(f"\n  --- FLUKE ({backbone}) ---")
        set_seed(42)
        fluke = FLUKEModel(
            model_name=backbone, embedding_dim=128,
            query_max_length=32, doc_max_length=180,
            use_tir=True, use_cqi=True, use_soft_topk=True,
            disc_scale=0.0, coverage_weight=0.0,
        )
        print(f"  Training ({len(triplets)} triplets, {num_epochs} epochs)...")
        train_model(fluke, triplets, num_epochs=num_epochs, batch_size=bs, lr=lr)
        fluke_metrics = eval_model(fluke, "fluke")
        print(f"  FLUKE:     nDCG@10={fluke_metrics['nDCG@10']:.4f}, "
              f"Recall@100={fluke_metrics['Recall@100']:.4f}")
        results[f"{ds_name}_fluke"] = fluke_metrics
        del fluke
        gc.collect()

        # Summary
        gap = fluke_metrics["nDCG@10"] - colbert_metrics["nDCG@10"]
        print(f"\n  Gap (FLUKE - ColBERTv2): nDCG@10 = {gap:+.4f}")

    # Overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ds_name in datasets:
        cm = results[f"{ds_name}_colbert"]
        fm = results[f"{ds_name}_fluke"]
        gap = fm["nDCG@10"] - cm["nDCG@10"]
        print(f"  {ds_name:12s}: ColBERTv2={cm['nDCG@10']:.4f}  FLUKE={fm['nDCG@10']:.4f}  gap={gap:+.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/real_beir_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/real_beir_results.json")


if __name__ == "__main__":
    run_real_beir_eval()
