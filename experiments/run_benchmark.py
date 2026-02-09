#!/usr/bin/env python3
"""Main experiment: compare FLUKE vs ColBERTv2 on BEIR benchmarks.

This script:
1. Loads BEIR datasets (SciFact, NFCorpus)
2. Trains both ColBERTv2 and FLUKE from the same pre-trained checkpoint
   on MS MARCO triplets (or evaluates zero-shot)
3. Evaluates on BEIR datasets
4. Reports nDCG@10, MAP@10, Recall@100 for both models
5. Runs ablation studies on FLUKE components

Usage:
    python experiments/run_benchmark.py --mode train_eval
    python experiments/run_benchmark.py --mode zero_shot
    python experiments/run_benchmark.py --mode ablation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluke.models.colbert import ColBERTModel
from fluke.models.fluke_model import FLUKEModel
from fluke.evaluation.benchmarks import BEIREvaluator
from fluke.training.trainer import train_model


def load_msmarco_triplets(max_triplets: int = 10000):
    """Load MS MARCO triplets for training from HuggingFace."""
    from datasets import load_dataset

    print(f"Loading MS MARCO triplets (max {max_triplets})...")
    ds = load_dataset(
        "sentence-transformers/msmarco-triplets",
        "triplet",
        split=f"train[:{max_triplets}]",
        trust_remote_code=True,
    )

    triplets = []
    for row in ds:
        query = row["anchor"]
        positive = row["positive"]
        negative = row["negative"]
        triplets.append((query, positive, negative))

    print(f"  Loaded {len(triplets)} triplets")
    return triplets


def run_zero_shot(
    datasets: list[str],
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
):
    """Zero-shot evaluation: no task-specific training."""
    results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        evaluator = BEIREvaluator(dataset_name)

        # Evaluate ColBERTv2
        print(f"\n--- ColBERTv2 (MaxSim) ---")
        colbert = ColBERTModel(model_name=model_name)
        colbert_metrics = evaluator.evaluate_model(
            colbert, model_type="colbert", batch_size=batch_size
        )
        print(f"ColBERTv2 results:")
        for k, v in sorted(colbert_metrics.items()):
            print(f"  {k}: {v:.4f}")

        # Evaluate FLUKE (all components)
        print(f"\n--- FLUKE (CQI + SoftTopK + TIR) ---")
        fluke = FLUKEModel(model_name=model_name, use_tir=True, use_cqi=True, use_soft_topk=True)
        fluke_metrics = evaluator.evaluate_model(
            fluke, model_type="fluke", batch_size=batch_size
        )
        print(f"FLUKE results:")
        for k, v in sorted(fluke_metrics.items()):
            print(f"  {k}: {v:.4f}")

        results[dataset_name] = {
            "colbert": colbert_metrics,
            "fluke": fluke_metrics,
        }

        # Cleanup
        del colbert, fluke
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def run_train_eval(
    datasets: list[str],
    model_name: str = "distilbert-base-uncased",
    num_triplets: int = 10000,
    num_epochs: int = 2,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
):
    """Train on MS MARCO, evaluate on BEIR."""
    # Load training data
    triplets = load_msmarco_triplets(max_triplets=num_triplets)

    results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        evaluator = BEIREvaluator(dataset_name)

        # --- ColBERTv2 ---
        print(f"\n--- Training ColBERTv2 ---")
        colbert = ColBERTModel(model_name=model_name)
        train_model(colbert, triplets, num_epochs=num_epochs, batch_size=train_batch_size)
        colbert.eval()

        print(f"\n--- Evaluating ColBERTv2 ---")
        colbert_metrics = evaluator.evaluate_model(
            colbert, model_type="colbert", batch_size=eval_batch_size
        )
        print(f"ColBERTv2 results:")
        for k, v in sorted(colbert_metrics.items()):
            print(f"  {k}: {v:.4f}")

        # --- FLUKE ---
        print(f"\n--- Training FLUKE ---")
        fluke = FLUKEModel(model_name=model_name, use_tir=True, use_cqi=True, use_soft_topk=True)
        train_model(fluke, triplets, num_epochs=num_epochs, batch_size=train_batch_size)
        fluke.eval()

        print(f"\n--- Evaluating FLUKE ---")
        fluke_metrics = evaluator.evaluate_model(
            fluke, model_type="fluke", batch_size=eval_batch_size
        )
        print(f"FLUKE results:")
        for k, v in sorted(fluke_metrics.items()):
            print(f"  {k}: {v:.4f}")

        results[dataset_name] = {
            "colbert": colbert_metrics,
            "fluke": fluke_metrics,
        }

        del colbert, fluke

    return results


def run_ablation(
    dataset_name: str = "scifact",
    model_name: str = "distilbert-base-uncased",
    num_triplets: int = 10000,
    num_epochs: int = 2,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
):
    """Ablation study: evaluate impact of each FLUKE component."""
    triplets = load_msmarco_triplets(max_triplets=num_triplets)
    evaluator = BEIREvaluator(dataset_name)

    configs = {
        "ColBERTv2 (baseline)": {"model_class": "colbert"},
        "FLUKE (CQI only)": {"use_cqi": True, "use_soft_topk": False, "use_tir": False},
        "FLUKE (SoftTopK only)": {"use_cqi": False, "use_soft_topk": True, "use_tir": False},
        "FLUKE (TIR only)": {"use_cqi": False, "use_soft_topk": False, "use_tir": True},
        "FLUKE (CQI + SoftTopK)": {"use_cqi": True, "use_soft_topk": True, "use_tir": False},
        "FLUKE (CQI + TIR)": {"use_cqi": True, "use_soft_topk": False, "use_tir": True},
        "FLUKE (all)": {"use_cqi": True, "use_soft_topk": True, "use_tir": True},
    }

    results = {}
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"{'='*60}")

        if config.get("model_class") == "colbert":
            model = ColBERTModel(model_name=model_name)
            model_type = "colbert"
        else:
            model = FLUKEModel(
                model_name=model_name,
                use_cqi=config["use_cqi"],
                use_soft_topk=config["use_soft_topk"],
                use_tir=config["use_tir"],
            )
            model_type = "fluke"

        train_model(model, triplets, num_epochs=num_epochs, batch_size=train_batch_size)
        model.eval()

        metrics = evaluator.evaluate_model(
            model, model_type=model_type, batch_size=eval_batch_size
        )
        print(f"Results:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")

        results[name] = metrics
        del model

    return results


def format_results_table(results: dict, title: str = "Results") -> str:
    """Format results as a readable ASCII table."""
    lines = [f"\n{title}", "=" * 80]

    if not results:
        return "\n".join(lines + ["No results."])

    # Check if results are nested by dataset
    first_val = next(iter(results.values()))
    if isinstance(first_val, dict) and any(isinstance(v, dict) for v in first_val.values()):
        # Nested: {dataset: {model: {metric: value}}}
        for dataset, models in results.items():
            lines.append(f"\nDataset: {dataset}")
            lines.append("-" * 60)
            header = f"{'Model':<30}"
            metrics = sorted(next(iter(models.values())).keys())
            for m in metrics:
                header += f" {m:>10}"
            lines.append(header)
            lines.append("-" * 60)
            for model_name, model_metrics in models.items():
                row = f"{model_name:<30}"
                for m in metrics:
                    row += f" {model_metrics.get(m, 0):>10.4f}"
                lines.append(row)
    else:
        # Flat: {config_name: {metric: value}}
        metrics = sorted(next(iter(results.values())).keys())
        header = f"{'Configuration':<35}"
        for m in metrics:
            header += f" {m:>10}"
        lines.append(header)
        lines.append("-" * 80)
        for config_name, config_metrics in results.items():
            row = f"{config_name:<35}"
            for m in metrics:
                row += f" {config_metrics.get(m, 0):>10.4f}"
            lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="FLUKE vs ColBERTv2 Benchmark")
    parser.add_argument(
        "--mode", choices=["zero_shot", "train_eval", "ablation"],
        default="train_eval", help="Evaluation mode",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["scifact", "nfcorpus"],
        help="BEIR datasets to evaluate on",
    )
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--num-triplets", type=int, default=10000)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/benchmark_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    start_time = time.time()

    if args.mode == "zero_shot":
        results = run_zero_shot(
            args.datasets, model_name=args.model, batch_size=args.eval_batch_size,
        )
    elif args.mode == "train_eval":
        results = run_train_eval(
            args.datasets,
            model_name=args.model,
            num_triplets=args.num_triplets,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
        )
    elif args.mode == "ablation":
        results = run_ablation(
            dataset_name=args.datasets[0],
            model_name=args.model,
            num_triplets=args.num_triplets,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
        )

    elapsed = time.time() - start_time
    print(f"\n\nTotal time: {elapsed:.1f}s")

    # Save results
    output_data = {
        "mode": args.mode,
        "model": args.model,
        "datasets": args.datasets,
        "results": results,
        "elapsed_seconds": elapsed,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.output}")

    # Print formatted table
    table = format_results_table(results, title=f"FLUKE vs ColBERTv2 ({args.mode})")
    print(table)


if __name__ == "__main__":
    main()
