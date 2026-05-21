#!/usr/bin/env python3
"""Grid search over disc_scale and coverage_weight for per-token disc approach.

Trains each model once, then evaluates with multiple parameter settings.
"""

import json
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluke.models.colbert import ColBERTModel
from fluke.models.fluke_model import FLUKEModel
from fluke.indexing.indexer import TokenEmbeddingIndex
from fluke.indexing.searcher import LatentSearcher
from fluke.training.trainer import train_model

from experiments.run_full_benchmark import (
    generate_synthetic_dataset,
    generate_lotte_style_dataset,
    evaluate_retrieval,
    set_seed,
)


def evaluate_with_params(model, corpus, queries, qrels, disc_scale, coverage_weight):
    """Evaluate FLUKE with specific disc_scale and coverage_weight."""
    old_disc = model.disc_scale
    old_cov = model.coverage_weight
    model.disc_scale = disc_scale
    model.coverage_weight = coverage_weight
    try:
        metrics = evaluate_retrieval(model, corpus, queries, qrels, "fluke")
    finally:
        model.disc_scale = old_disc
        model.coverage_weight = old_cov
    return metrics


def run_grid_search():
    print("=" * 70)
    print("GRID SEARCH: Per-token Discriminativeness")
    print("=" * 70)

    disc_scale_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    coverage_weight_values = [0.0, 0.1, 0.2, 0.3]

    num_epochs = 6
    lr = 5e-5
    batch_size = 16
    results = {}

    # ---- BEIR ----
    print("\n--- BEIR-style benchmark ---")
    set_seed(42)
    beir_corpus, beir_queries, beir_qrels, beir_triplets = generate_synthetic_dataset(
        n_topics=50, docs_per_topic=20, noise_docs=200, queries_per_topic=5,
    )
    beir_train = beir_triplets[:min(len(beir_triplets), 3000)]
    print(f"  Corpus: {len(beir_corpus)}, Queries: {len(beir_queries)}, Triplets: {len(beir_train)}")

    # Train ColBERTv2 baseline
    print("\n  Training ColBERTv2 baseline...")
    set_seed(42)
    colbert = ColBERTModel(model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64)
    train_model(colbert, beir_train, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
    beir_colbert = evaluate_retrieval(colbert, beir_corpus, beir_queries, beir_qrels, "colbert")
    print(f"    ColBERTv2: nDCG@10={beir_colbert['nDCG@10']:.4f}")
    results["beir_colbert"] = beir_colbert
    del colbert

    # Train FLUKE (with disc_scale=1.5 default during training)
    print("\n  Training FLUKE...")
    set_seed(42)
    fluke_beir = FLUKEModel(
        model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64,
        use_tir=True, use_cqi=True, use_soft_topk=True,
    )
    train_model(fluke_beir, beir_train, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

    # Grid search over disc_scale and coverage_weight at eval time
    print("\n  Grid search on BEIR...")
    beir_grid = {}
    for ds in disc_scale_values:
        for cw in coverage_weight_values:
            key = f"ds={ds}_cw={cw}"
            metrics = evaluate_with_params(fluke_beir, beir_corpus, beir_queries, beir_qrels, ds, cw)
            beir_grid[key] = metrics
            gap = metrics["nDCG@10"] - beir_colbert["nDCG@10"]
            print(f"    disc_scale={ds:.1f}, cov_weight={cw:.1f}: nDCG@10={metrics['nDCG@10']:.4f} (gap={gap:+.4f})")
    results["beir_grid"] = beir_grid
    del fluke_beir

    # ---- LoTTE ----
    lotte_domains = ["science", "technology", "writing"]
    lotte_colbert_results = {}
    lotte_grids = {}

    for domain in lotte_domains:
        print(f"\n--- LoTTE-style: {domain} ---")
        set_seed(42)
        corpus, queries, qrels, triplets = generate_lotte_style_dataset(domain_name=domain, seed=42)
        train_triplets = triplets[:min(len(triplets), 3000)]

        # ColBERTv2 baseline
        print(f"  Training ColBERTv2 for {domain}...")
        set_seed(42)
        colbert = ColBERTModel(model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64)
        train_model(colbert, train_triplets, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
        colbert_metrics = evaluate_retrieval(colbert, corpus, queries, qrels, "colbert")
        print(f"    ColBERTv2: nDCG@10={colbert_metrics['nDCG@10']:.4f}")
        lotte_colbert_results[domain] = colbert_metrics
        del colbert

        # FLUKE
        print(f"  Training FLUKE for {domain}...")
        set_seed(42)
        fluke = FLUKEModel(
            model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64,
            use_tir=True, use_cqi=True, use_soft_topk=True,
        )
        train_model(fluke, train_triplets, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

        # Grid search
        print(f"  Grid search on {domain}...")
        domain_grid = {}
        for ds in disc_scale_values:
            for cw in coverage_weight_values:
                key = f"ds={ds}_cw={cw}"
                metrics = evaluate_with_params(fluke, corpus, queries, qrels, ds, cw)
                domain_grid[key] = metrics
                gap = metrics["nDCG@10"] - colbert_metrics["nDCG@10"]
                print(f"    disc_scale={ds:.1f}, cov_weight={cw:.1f}: nDCG@10={metrics['nDCG@10']:.4f} (gap={gap:+.4f})")
        lotte_grids[domain] = domain_grid
        del fluke

    results["lotte_colbert"] = lotte_colbert_results
    results["lotte_grids"] = lotte_grids

    # ---- Find best combined setting ----
    print("\n" + "=" * 70)
    print("FINDING BEST COMBINED SETTING")
    print("=" * 70)

    best_score = -999
    best_key = None
    for ds in disc_scale_values:
        for cw in coverage_weight_values:
            key = f"ds={ds}_cw={cw}"
            beir_gap = beir_grid[key]["nDCG@10"] - results["beir_colbert"]["nDCG@10"]

            lotte_gaps = []
            for domain in lotte_domains:
                lotte_gap = lotte_grids[domain][key]["nDCG@10"] - lotte_colbert_results[domain]["nDCG@10"]
                lotte_gaps.append(lotte_gap)
            avg_lotte_gap = np.mean(lotte_gaps)
            min_lotte_gap = min(lotte_gaps)

            # Combined score: want both gaps positive, penalize if any gap is negative
            combined = beir_gap + avg_lotte_gap + min_lotte_gap * 0.5
            if min_lotte_gap < 0:
                combined -= 0.5  # heavy penalty for any negative gap

            print(f"  {key}: BEIR gap={beir_gap:+.4f}, LoTTE avg gap={avg_lotte_gap:+.4f}, "
                  f"LoTTE min gap={min_lotte_gap:+.4f}, combined={combined:+.4f}")

            if combined > best_score:
                best_score = combined
                best_key = key

    print(f"\n  BEST: {best_key} (combined score={best_score:+.4f})")
    results["best_setting"] = best_key
    results["best_combined_score"] = best_score

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/grid_search_pertoken_disc.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_grid_search()
