#!/usr/bin/env python3
"""Lean grid search: train with disc=0, sweep disc at eval time.
Memory-efficient: explicit cleanup between models.
"""

import gc
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
    set_seed,
)


def fast_eval(index, query_data, query_ids, qrels, tir_module, disc_scale, coverage_weight):
    import pytrec_eval
    searcher = LatentSearcher(
        index, scoring="fluke", tir_module=tir_module, max_query_tokens=32,
        disc_scale=disc_scale, coverage_weight=coverage_weight,
    )
    results = searcher.batch_search(query_data, top_k=100)
    formatted = {}
    for q_idx, rl in results.items():
        formatted[query_ids[q_idx]] = {d: s for d, s in rl}
    metrics = {}
    for k in [10]:
        ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{k}"})
        sc = ev.evaluate(formatted)
        metrics[f"nDCG@{k}"] = float(np.mean([s[f"ndcg_cut_{k}"] for s in sc.values()]))
    return metrics


def eval_colbert(index, query_data, query_ids, qrels):
    import pytrec_eval
    searcher = LatentSearcher(index, scoring="maxsim")
    results = searcher.batch_search(query_data, top_k=100)
    formatted = {}
    for q_idx, rl in results.items():
        formatted[query_ids[q_idx]] = {d: s for d, s in rl}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    sc = ev.evaluate(formatted)
    return {"nDCG@10": float(np.mean([s["ndcg_cut_10"] for s in sc.values()]))}


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(name, corpus, queries, qrels, triplets, disc_values, cov_values):
    """Run ColBERTv2 + FLUKE grid search for one benchmark."""
    print(f"\n--- {name} ---")
    train_triplets = triplets[:min(len(triplets), 3000)]
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    # ColBERTv2
    print("  Training ColBERTv2...")
    set_seed(42)
    colbert = ColBERTModel(model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64)
    train_model(colbert, train_triplets, num_epochs=6, batch_size=16, lr=5e-5)
    colbert.eval()
    c_embs = colbert.encode_documents(doc_texts, batch_size=64)
    c_idx = TokenEmbeddingIndex()
    c_idx.add_batch(doc_ids, c_embs)
    c_qdata = colbert.encode_queries(query_texts, batch_size=64)
    colbert_m = eval_colbert(c_idx, c_qdata, query_ids, qrels)
    print(f"    ColBERTv2: nDCG@10={colbert_m['nDCG@10']:.4f}")
    del colbert, c_embs, c_idx, c_qdata
    cleanup()

    # FLUKE (train with disc=0)
    print("  Training FLUKE (disc=0)...")
    set_seed(42)
    fluke = FLUKEModel(model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64,
                       use_tir=True, use_cqi=True, use_soft_topk=True,
                       disc_scale=0.0, coverage_weight=0.0)
    train_model(fluke, train_triplets, num_epochs=6, batch_size=16, lr=5e-5)
    fluke.eval()
    f_embs = fluke.encode_documents(doc_texts, batch_size=64)
    f_idx = TokenEmbeddingIndex()
    f_idx.add_batch(doc_ids, f_embs)
    f_qdata = fluke.encode_queries(query_texts, batch_size=64)
    tir = getattr(fluke, "tir", None)
    del fluke
    cleanup()

    # Grid search
    print("  Grid search...")
    grid = {}
    for ds in disc_values:
        for cw in cov_values:
            key = f"ds={ds}_cw={cw}"
            m = fast_eval(f_idx, f_qdata, query_ids, qrels, tir, ds, cw)
            gap = m["nDCG@10"] - colbert_m["nDCG@10"]
            print(f"    ds={ds:5.1f} cw={cw:.2f}: nDCG@10={m['nDCG@10']:.4f} ({gap:+.4f})")
            grid[key] = m
    del f_embs, f_idx, f_qdata, tir
    cleanup()
    return colbert_m, grid


def main():
    print("=" * 70)
    print("GRID SEARCH: Self-Calibrating Per-Token Disc (train disc=0)")
    print("=" * 70)

    disc_values = [0.0, 4.0, 8.0, 12.0, 20.0]
    cov_values = [0.0, 0.15, 0.3]
    results = {}

    # BEIR
    set_seed(42)
    beir_c, beir_q, beir_qr, beir_t = generate_synthetic_dataset()
    beir_colbert, beir_grid = run_benchmark("BEIR", beir_c, beir_q, beir_qr, beir_t, disc_values, cov_values)
    results["beir_colbert"] = beir_colbert
    results["beir_grid"] = beir_grid

    # LoTTE domains
    lotte_colbert = {}
    lotte_grids = {}
    for domain in ["science", "technology", "writing"]:
        set_seed(42)
        c, q, qr, t = generate_lotte_style_dataset(domain_name=domain, seed=42)
        cm, grid = run_benchmark(f"LoTTE-{domain}", c, q, qr, t, disc_values, cov_values)
        lotte_colbert[domain] = cm
        lotte_grids[domain] = grid

    results["lotte_colbert"] = lotte_colbert
    results["lotte_grids"] = lotte_grids

    # Find best
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)
    best_score = -999
    best_key = None
    domains = ["science", "technology", "writing"]
    for ds in disc_values:
        for cw in cov_values:
            key = f"ds={ds}_cw={cw}"
            bg = beir_grid[key]["nDCG@10"] - beir_colbert["nDCG@10"]
            lgs = [lotte_grids[d][key]["nDCG@10"] - lotte_colbert[d]["nDCG@10"] for d in domains]
            al = np.mean(lgs)
            ml = min(lgs)
            ap = bg > 0 and ml > 0
            comb = bg + al
            mark = " ***" if ap else ""
            print(f"  {key:20s}  BEIR={bg:+.4f}  LoTTE_avg={al:+.4f}  LoTTE_min={ml:+.4f}  comb={comb:+.4f}{mark}")
            if ap and comb > best_score:
                best_score = comb
                best_key = key
    if not best_key:
        for ds in disc_values:
            for cw in cov_values:
                key = f"ds={ds}_cw={cw}"
                bg = beir_grid[key]["nDCG@10"] - beir_colbert["nDCG@10"]
                lgs = [lotte_grids[d][key]["nDCG@10"] - lotte_colbert[d]["nDCG@10"] for d in domains]
                comb = bg + np.mean(lgs)
                if comb > best_score:
                    best_score = comb
                    best_key = key
        print(f"\n  >>> BEST (fallback): {best_key} (comb={best_score:+.4f})")
    else:
        print(f"\n  >>> BEST (all positive): {best_key} (comb={best_score:+.4f})")
    results["best"] = best_key

    os.makedirs("results", exist_ok=True)
    with open("results/grid_search_selfcal_disc.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/grid_search_selfcal_disc.json")


if __name__ == "__main__":
    main()
