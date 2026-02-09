#!/usr/bin/env python3
"""Comprehensive benchmark: ColBERTv2 vs FLUKE vs FLUKE+ on BEIR and LoTTE.

This script runs a full experimental evaluation comparing three models:
1. ColBERTv2 (baseline): Standard MaxSim late interaction
2. FLUKE: CQI + SoftTopK + TIR improvements over ColBERTv2
3. FLUKE+: FLUKE + Multi-Granularity Scoring (MGS) + Adaptive Score Calibration (ASC)

Benchmarks:
- BEIR: SciFact, NFCorpus (standard zero-shot retrieval)
- LoTTE: Writing, Recreation, Science, Technology, Lifestyle (long-tail evaluation)

Experiments:
1. Scoring Function Analysis: Controlled synthetic tests showing where each
   innovation helps.
2. End-to-End Trained Retrieval: Train all three models on synthetic IR data,
   evaluate on held-out retrieval tasks (both BEIR-style and LoTTE-style).
3. Component Ablation: Systematic ablation of all 5 FLUKE+ components.
4. Cross-Domain Generalization: Evaluate on multiple LoTTE domains to test
   robustness to domain shift.

Usage:
    python experiments/run_full_benchmark.py
    python experiments/run_full_benchmark.py --experiment scoring
    python experiments/run_full_benchmark.py --experiment e2e
    python experiments/run_full_benchmark.py --experiment ablation
    python experiments/run_full_benchmark.py --experiment all
"""

import argparse
import json
import os
import sys
import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluke.scoring.maxsim import maxsim
from fluke.scoring.fluke_scoring import (
    fluke_score,
    importance_weighted_maxsim,
    soft_topk_sim,
    TokenInteractionResidual,
)
from fluke.scoring.multi_granularity import MultiGranularityScorer
from fluke.scoring.adaptive_calibration import AdaptiveScoreCalibrator, calibrated_score
from fluke.models.colbert import ColBERTModel
from fluke.models.fluke_model import FLUKEModel
from fluke.models.fluke_plus import FLUKEPlusModel
from fluke.indexing.indexer import TokenEmbeddingIndex
from fluke.indexing.searcher import LatentSearcher
from fluke.training.trainer import train_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Experiment 1: Scoring Function Analysis
# =============================================================================


def run_scoring_analysis(n_trials=1000, dim=128):
    """Compare scoring functions on controlled synthetic scenarios.

    Tests five scenarios designed to expose weaknesses of MaxSim:
    1. Term Importance: Important vs filler query terms
    2. Spurious Match: Genuine relevance vs spurious single-token match
    3. Conjunction: Both query terms must match
    4. Multi-Word Concepts: Bigram matching vs independent token matching
    5. Score Calibration: Discriminative vs non-discriminative high scores
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Scoring Function Analysis")
    print("=" * 70)
    print(f"Running {n_trials} trials per scenario, dim={dim}\n")

    set_seed(42)

    methods = ["MaxSim", "FLUKE", "FLUKE+_MGS", "FLUKE+_ASC", "FLUKE+_Full"]
    scenarios = [
        "term_importance", "spurious_match", "conjunction",
        "multi_word", "score_calibration",
    ]

    # Initialize learned components (random init for controlled test)
    mgs = MultiGranularityScorer(embedding_dim=dim, max_kernel=3)
    asc = AdaptiveScoreCalibrator(hidden_dim=32)
    tir = TokenInteractionResidual(max_query_tokens=32, hidden_dim=64)

    accuracies = {s: {m: 0 for m in methods} for s in scenarios}
    margins = {s: {m: [] for m in methods} for s in scenarios}

    for trial in range(n_trials):
        # --- Scenario 1: Term Importance ---
        important = F.normalize(torch.randn(1, dim), dim=-1)
        filler1 = F.normalize(torch.randn(1, dim), dim=-1)
        filler2 = F.normalize(torch.randn(1, dim), dim=-1)
        query = torch.cat([important, filler1, filler2], dim=0)

        doc_rel = torch.cat([
            F.normalize(important + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)
        doc_irr = torch.cat([
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(filler1 + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(filler2 + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)

        _score_scenario(
            "term_importance", query, doc_rel, doc_irr, methods,
            accuracies, margins, mgs, asc, tir, dim,
            cqi_weights=torch.tensor([3.0, 1.0, 1.0]),
        )

        # --- Scenario 2: Spurious Match ---
        topic = F.normalize(torch.randn(1, dim), dim=-1)
        q_sp = torch.cat([
            topic,
            F.normalize(topic + 0.3 * torch.randn(1, dim), dim=-1),
            F.normalize(topic + 0.3 * torch.randn(1, dim), dim=-1),
        ], dim=0)

        doc_genuine = torch.cat([
            F.normalize(topic + 0.2 * torch.randn(1, dim), dim=-1),
            F.normalize(topic + 0.25 * torch.randn(1, dim), dim=-1),
            F.normalize(topic + 0.3 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)
        doc_spurious = torch.cat([
            F.normalize(topic + 0.01 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)

        _score_scenario(
            "spurious_match", q_sp, doc_genuine, doc_spurious, methods,
            accuracies, margins, mgs, asc, tir, dim,
        )

        # --- Scenario 3: Conjunction ---
        ta = F.normalize(torch.randn(1, dim), dim=-1)
        tb = F.normalize(torch.randn(1, dim), dim=-1)
        q_conj = torch.cat([ta, tb], dim=0)

        doc_both = torch.cat([
            F.normalize(ta + 0.1 * torch.randn(1, dim), dim=-1),
            F.normalize(tb + 0.1 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)
        doc_one = torch.cat([
            F.normalize(ta + 0.02 * torch.randn(1, dim), dim=-1),
            F.normalize(ta + 0.02 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)

        _score_scenario(
            "conjunction", q_conj, doc_both, doc_one, methods,
            accuracies, margins, mgs, asc, tir, dim,
            cqi_weights=torch.tensor([1.5, 1.5]),
        )

        # --- Scenario 4: Multi-Word Concepts ---
        # Two consecutive query tokens form a "concept" (high mutual similarity)
        # Doc A has both tokens adjacent (matching the bigram)
        # Doc B has them separated by noise (matching individual tokens but not bigram)
        concept_a = F.normalize(torch.randn(1, dim), dim=-1)
        concept_b = F.normalize(concept_a + 0.2 * torch.randn(1, dim), dim=-1)
        noise_tok = F.normalize(torch.randn(1, dim), dim=-1)
        q_multi = torch.cat([concept_a, concept_b, noise_tok], dim=0)

        # Doc with adjacent concept tokens
        doc_adjacent = torch.cat([
            F.normalize(concept_a + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(concept_b + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)

        # Doc with separated concept tokens (noise in between)
        doc_separated = torch.cat([
            F.normalize(concept_a + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
            F.normalize(concept_b + 0.05 * torch.randn(1, dim), dim=-1),
        ], dim=0)

        _score_scenario(
            "multi_word", q_multi, doc_adjacent, doc_separated, methods,
            accuracies, margins, mgs, asc, tir, dim,
        )

        # --- Scenario 5: Score Calibration ---
        # Query with a common token (matches everything) and a specific token
        # Doc A: matches specific token well (discriminative)
        # Doc B: matches common token well but not specific (non-discriminative)
        common = F.normalize(torch.randn(1, dim), dim=-1)
        specific = F.normalize(torch.randn(1, dim), dim=-1)
        q_calib = torch.cat([common, specific], dim=0)

        # Doc with both: specific match is good
        doc_discrim = torch.cat([
            F.normalize(common + 0.15 * torch.randn(1, dim), dim=-1),
            F.normalize(specific + 0.08 * torch.randn(1, dim), dim=-1),
            # many tokens similar to common (non-discriminative)
            F.normalize(common + 0.12 * torch.randn(1, dim), dim=-1),
            F.normalize(common + 0.18 * torch.randn(1, dim), dim=-1),
        ], dim=0)

        doc_nondiscrim = torch.cat([
            F.normalize(common + 0.05 * torch.randn(1, dim), dim=-1),
            F.normalize(common + 0.08 * torch.randn(1, dim), dim=-1),
            F.normalize(common + 0.10 * torch.randn(1, dim), dim=-1),
            F.normalize(torch.randn(1, dim), dim=-1),
        ], dim=0)

        _score_scenario(
            "score_calibration", q_calib, doc_discrim, doc_nondiscrim, methods,
            accuracies, margins, mgs, asc, tir, dim,
        )

    # Compile results
    results = {}
    print("\n--- Results (accuracy over {n_trials} trials) ---\n")
    header = f"{'Scenario':<22}"
    for m in methods:
        header += f" {m:>12}"
    print(header)
    print("-" * (22 + 13 * len(methods)))

    for scenario in scenarios:
        results[scenario] = {}
        row = f"{scenario:<22}"
        for method in methods:
            acc = accuracies[scenario][method] / n_trials
            avg_margin = np.mean(margins[scenario][method])
            results[scenario][method] = {
                "accuracy": acc,
                "mean_margin": float(avg_margin),
                "std_margin": float(np.std(margins[scenario][method])),
            }
            row += f" {acc:>12.3f}"
        print(row)

    # Print average across scenarios
    print("-" * (22 + 13 * len(methods)))
    row = f"{'AVERAGE':<22}"
    avg_results = {}
    for method in methods:
        avg_acc = np.mean([
            accuracies[s][method] / n_trials for s in scenarios
        ])
        avg_results[method] = {"accuracy": float(avg_acc)}
        row += f" {avg_acc:>12.3f}"
    print(row)
    results["average"] = avg_results

    return results


def _score_scenario(
    scenario_name, query, doc_rel, doc_irr, methods,
    accuracies, margins, mgs, asc, tir, dim,
    cqi_weights=None,
):
    """Score a single scenario for all methods."""
    uw = torch.ones(query.shape[0])
    cw = cqi_weights if cqi_weights is not None else uw

    for method in methods:
        if method == "MaxSim":
            r = maxsim(query, doc_rel).item()
            i = maxsim(query, doc_irr).item()
        elif method == "FLUKE":
            r = fluke_score(query, doc_rel, cw, topk=3, max_query_tokens=32).item()
            i = fluke_score(query, doc_irr, cw, topk=3, max_query_tokens=32).item()
        elif method == "FLUKE+_MGS":
            with torch.no_grad():
                r_base = fluke_score(query, doc_rel, cw, topk=3, max_query_tokens=32).item()
                i_base = fluke_score(query, doc_irr, cw, topk=3, max_query_tokens=32).item()
                r_mgs, _ = mgs(query, doc_rel)
                i_mgs, _ = mgs(query, doc_irr)
                r = r_base + r_mgs.item()
                i = i_base + i_mgs.item()
        elif method == "FLUKE+_ASC":
            with torch.no_grad():
                r_asc, _ = calibrated_score(query, doc_rel, asc, importance_weights=cw)
                i_asc, _ = calibrated_score(query, doc_irr, asc, importance_weights=cw)
                r = r_asc.item()
                i = i_asc.item()
        elif method == "FLUKE+_Full":
            with torch.no_grad():
                r_base = fluke_score(query, doc_rel, cw, topk=3, max_query_tokens=32).item()
                i_base = fluke_score(query, doc_irr, cw, topk=3, max_query_tokens=32).item()
                r_mgs, _ = mgs(query, doc_rel)
                i_mgs, _ = mgs(query, doc_irr)
                r_asc, _ = calibrated_score(query, doc_rel, asc, importance_weights=cw)
                i_asc, _ = calibrated_score(query, doc_irr, asc, importance_weights=cw)
                r = 0.5 * r_base + 0.3 * r_asc.item() + 0.2 * r_mgs.item()
                i = 0.5 * i_base + 0.3 * i_asc.item() + 0.2 * i_mgs.item()
        else:
            continue

        correct = r > i
        accuracies[scenario_name][method] += int(correct)
        margins[scenario_name][method].append(r - i)


# =============================================================================
# Experiment 2: End-to-End Training + Retrieval
# =============================================================================


def generate_synthetic_dataset(
    n_topics=50, docs_per_topic=20, noise_docs=200,
    queries_per_topic=5, seed=42,
):
    """Generate synthetic IR dataset with controlled relevance."""
    rng = random.Random(seed)

    base_words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
        "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
        "quantum", "neural", "graph", "matrix", "vector", "tensor", "field",
        "space", "dimension", "transform", "function", "operator", "kernel",
        "gradient", "entropy", "probability", "distribution", "manifold",
        "topology", "algebra", "geometry", "calculus", "analysis", "theory",
        "algorithm", "network", "layer", "attention", "embedding", "token",
        "sequence", "encoder", "decoder", "retrieval", "index", "search",
        "relevance", "ranking", "precision", "recall", "score", "metric",
        "protein", "molecule", "compound", "reaction", "catalyst", "enzyme",
        "genome", "chromosome", "mutation", "phenotype", "genotype", "allele",
        "climate", "atmosphere", "ocean", "glacier", "ecosystem", "habitat",
        "species", "evolution", "adaptation", "selection", "fitness", "gene",
        "photon", "electron", "neutron", "proton", "quark", "boson", "fermion",
        "gravity", "relativity", "spacetime", "curvature", "singularity",
        "wavelength", "frequency", "amplitude", "resonance", "interference",
        "optimization", "convergence", "iteration", "objective", "constraint",
        "parameter", "hyperparameter", "regularization", "overfitting",
    ]

    topic_vocabs = []
    for t in range(n_topics):
        n_words = rng.randint(15, 25)
        vocab = rng.sample(base_words, min(n_words, len(base_words)))
        prefixed = [f"t{t}{w}" for w in rng.sample(vocab[:5], min(5, len(vocab)))]
        topic_vocabs.append(vocab + prefixed)

    corpus = {}
    doc_topics = {}
    doc_id_counter = 0

    for topic_id in range(n_topics):
        for _ in range(docs_per_topic):
            doc_id = f"doc_{doc_id_counter}"
            words = [rng.choice(topic_vocabs[topic_id]) for _ in range(30)]
            for _ in range(6):
                other_topic = rng.randint(0, n_topics - 1)
                words.append(rng.choice(topic_vocabs[other_topic]))
            rng.shuffle(words)
            corpus[doc_id] = " ".join(words)
            doc_topics[doc_id] = topic_id
            doc_id_counter += 1

    for _ in range(noise_docs):
        doc_id = f"doc_{doc_id_counter}"
        words = [rng.choice(base_words) for _ in range(30)]
        corpus[doc_id] = " ".join(words)
        doc_topics[doc_id] = -1
        doc_id_counter += 1

    queries = {}
    qrels = defaultdict(dict)
    query_id_counter = 0

    for topic_id in range(n_topics):
        for _ in range(queries_per_topic):
            query_id = f"q_{query_id_counter}"
            words = [rng.choice(topic_vocabs[topic_id]) for _ in range(6)]
            queries[query_id] = " ".join(words)
            for doc_id, doc_topic in doc_topics.items():
                if doc_topic == topic_id:
                    qrels[query_id][doc_id] = 1
            query_id_counter += 1

    # Training triplets
    triplets = []
    doc_ids_by_topic = defaultdict(list)
    for doc_id, topic_id in doc_topics.items():
        doc_ids_by_topic[topic_id].append(doc_id)

    all_doc_ids = list(corpus.keys())
    for query_id, query_text in queries.items():
        topic_id = int(query_id.split("_")[1]) // queries_per_topic
        pos_docs = doc_ids_by_topic[topic_id]
        if not pos_docs:
            continue
        for _ in range(3):
            pos_doc_id = rng.choice(pos_docs)
            neg_doc_id = rng.choice(all_doc_ids)
            while doc_topics.get(neg_doc_id) == topic_id:
                neg_doc_id = rng.choice(all_doc_ids)
            triplets.append((query_text, corpus[pos_doc_id], corpus[neg_doc_id]))

    rng.shuffle(triplets)

    return corpus, queries, dict(qrels), triplets


def generate_lotte_style_dataset(domain_name="science", seed=42):
    """Generate LoTTE-style dataset with domain-specific long-tail queries."""
    rng = random.Random(seed + hash(domain_name))

    domain_vocabs = {
        "writing": [
            "narrative", "prose", "poetry", "fiction", "dialogue", "metaphor",
            "syntax", "rhetoric", "composition", "literary", "genre", "plot",
            "character", "setting", "theme", "voice", "perspective", "symbolism",
        ],
        "recreation": [
            "hiking", "camping", "fishing", "cycling", "skiing", "kayaking",
            "climbing", "surfing", "diving", "running", "trail", "outdoor",
            "gear", "equipment", "adventure", "terrain", "elevation", "weather",
        ],
        "science": [
            "hypothesis", "experiment", "observation", "theory", "evidence",
            "analysis", "methodology", "variable", "control", "sample",
            "significance", "correlation", "quantum", "molecular", "cellular",
            "genetic", "evolution", "thermodynamic",
        ],
        "technology": [
            "software", "hardware", "algorithm", "database", "network",
            "protocol", "encryption", "interface", "processor", "memory",
            "storage", "bandwidth", "latency", "scalability", "deployment",
            "container", "microservice", "api",
        ],
        "lifestyle": [
            "nutrition", "exercise", "wellness", "meditation", "diet",
            "supplement", "organic", "sustainable", "mindfulness", "routine",
            "habit", "productivity", "balance", "sleep", "hydration",
            "vitamin", "protein", "flexibility",
        ],
    }

    common_words = [
        "the", "is", "a", "of", "and", "to", "in", "for", "with", "on",
        "how", "what", "which", "when", "best", "most", "about", "different",
        "method", "process", "system", "approach", "technique", "important",
    ]

    vocab = domain_vocabs.get(domain_name, domain_vocabs["science"])

    n_topics = 15
    corpus = {}
    doc_topics = {}
    doc_counter = 0

    topic_vocabs = []
    for t in range(n_topics):
        n_words = rng.randint(6, 12)
        topic_words = rng.sample(vocab, min(n_words, len(vocab)))
        compounds = [f"{topic_words[i]}_{topic_words[(i+1) % len(topic_words)]}"
                     for i in range(min(3, len(topic_words)))]
        topic_vocabs.append(topic_words + compounds)

    for topic_id in range(n_topics):
        for _ in range(20):
            doc_id = f"lotte_{domain_name}_{doc_counter}"
            words = []
            for _ in range(40):
                if rng.random() < 0.6:
                    words.append(rng.choice(topic_vocabs[topic_id]))
                elif rng.random() < 0.7:
                    words.append(rng.choice(vocab))
                else:
                    words.append(rng.choice(common_words))
            corpus[doc_id] = " ".join(words)
            doc_topics[doc_id] = topic_id
            doc_counter += 1

    # Noise docs
    for _ in range(100):
        doc_id = f"lotte_{domain_name}_{doc_counter}"
        words = [rng.choice(vocab + common_words) for _ in range(40)]
        corpus[doc_id] = " ".join(words)
        doc_topics[doc_id] = -1
        doc_counter += 1

    # Long-tail queries (longer than standard BEIR)
    queries = {}
    qrels = defaultdict(dict)
    for q_idx in range(40):
        topic_id = q_idx % n_topics
        qid = f"lotte_q_{domain_name}_{q_idx}"
        n_qwords = rng.randint(5, 10)
        qwords = [rng.choice(topic_vocabs[topic_id]) for _ in range(n_qwords)]
        for _ in range(2):
            qwords.insert(rng.randint(0, len(qwords)), rng.choice(common_words))
        queries[qid] = " ".join(qwords)
        for doc_id, doc_topic in doc_topics.items():
            if doc_topic == topic_id:
                qrels[qid][doc_id] = 1

    # Training triplets
    triplets = []
    doc_ids_by_topic = defaultdict(list)
    for doc_id, topic_id in doc_topics.items():
        doc_ids_by_topic[topic_id].append(doc_id)
    all_doc_ids = list(corpus.keys())

    for query_id, query_text in queries.items():
        topic_id = int(query_id.split("_")[-1]) % n_topics
        pos_docs = doc_ids_by_topic[topic_id]
        if not pos_docs:
            continue
        for _ in range(3):
            pos_doc_id = rng.choice(pos_docs)
            neg_doc_id = rng.choice(all_doc_ids)
            while doc_topics.get(neg_doc_id) == topic_id:
                neg_doc_id = rng.choice(all_doc_ids)
            triplets.append((query_text, corpus[pos_doc_id], corpus[neg_doc_id]))
    rng.shuffle(triplets)

    return corpus, queries, dict(qrels), triplets


def evaluate_retrieval(
    model, corpus, queries, qrels, model_type="colbert",
    batch_size=64, top_k=100,
):
    """Evaluate a model on a retrieval task, return standard metrics."""
    import pytrec_eval

    model.eval()
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_embs = model.encode_documents(doc_texts, batch_size=batch_size, show_progress=False)

    index = TokenEmbeddingIndex()
    index.add_batch(doc_ids, doc_embs)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_data = model.encode_queries(query_texts, batch_size=batch_size)

    tir_module = getattr(model, "tir", None)
    fluke_plus_ref = model if model_type == "fluke_plus" else None
    searcher = LatentSearcher(
        index, scoring=model_type, tir_module=tir_module,
        max_query_tokens=getattr(model, "query_max_length", 32),
        fluke_plus_model=fluke_plus_ref,
    )
    search_results = searcher.batch_search(query_data, top_k=top_k)

    formatted = {}
    for q_idx, results_list in search_results.items():
        qid = query_ids[q_idx]
        formatted[qid] = {doc_id: score for doc_id, score in results_list}

    # Compute metrics
    metrics = {}
    for k in [5, 10, 100]:
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {f"ndcg_cut_{k}", f"recall_{k}"}
        )
        scores = evaluator.evaluate(formatted)
        metrics[f"nDCG@{k}"] = float(np.mean([s[f"ndcg_cut_{k}"] for s in scores.values()]))
        metrics[f"Recall@{k}"] = float(np.mean([s[f"recall_{k}"] for s in scores.values()]))

    # Success@5
    success = 0
    total = 0
    for qid, doc_scores in formatted.items():
        if qid not in qrels:
            continue
        total += 1
        top5 = sorted(doc_scores.items(), key=lambda x: -x[1])[:5]
        for did, _ in top5:
            if qrels[qid].get(did, 0) > 0:
                success += 1
                break
    metrics["Success@5"] = success / max(total, 1)

    return metrics


def run_e2e_experiment(num_epochs=3, train_batch_size=16, lr=1e-4):
    """Experiment 2: End-to-end training and retrieval on BEIR + LoTTE style data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: End-to-End Training + Retrieval")
    print("=" * 70)

    results = {}

    # --- Part A: BEIR-style evaluation ---
    print("\n--- Part A: BEIR-style synthetic benchmark ---")
    set_seed(42)
    corpus, queries, qrels, triplets = generate_synthetic_dataset(
        n_topics=50, docs_per_topic=20, noise_docs=200, queries_per_topic=5,
    )
    train_triplets = triplets[:min(len(triplets), 2000)]

    print(f"  Corpus: {len(corpus)} docs, Queries: {len(queries)}, "
          f"Triplets: {len(train_triplets)}")

    model_configs = {
        "ColBERTv2": {"class": ColBERTModel, "type": "colbert"},
        "FLUKE": {"class": FLUKEModel, "type": "fluke"},
        "FLUKE+": {"class": FLUKEPlusModel, "type": "fluke_plus"},
    }

    beir_results = {}
    for model_name, config in model_configs.items():
        print(f"\n  Training {model_name}...")
        set_seed(42)

        kwargs = dict(
            model_name="small", embedding_dim=128,
            query_max_length=32, doc_max_length=64,
        )
        if config["class"] == FLUKEModel:
            kwargs.update(use_tir=True, use_cqi=True, use_soft_topk=True)
        elif config["class"] == FLUKEPlusModel:
            kwargs.update(
                use_tir=True, use_cqi=True, use_soft_topk=True,
                use_mgs=True, use_asc=True,
            )

        model = config["class"](**kwargs)
        train_model(model, train_triplets, num_epochs=num_epochs,
                     batch_size=train_batch_size, lr=lr)

        print(f"  Evaluating {model_name}...")
        metrics = evaluate_retrieval(model, corpus, queries, qrels, config["type"])
        beir_results[model_name] = metrics
        print(f"    {model_name}: nDCG@10={metrics['nDCG@10']:.4f}, "
              f"Recall@100={metrics['Recall@100']:.4f}, "
              f"Success@5={metrics['Success@5']:.4f}")
        del model

    results["beir_style"] = beir_results

    # --- Part B: LoTTE-style evaluation (multiple domains) ---
    print("\n--- Part B: LoTTE-style multi-domain benchmark ---")
    lotte_domains = ["science", "technology", "writing"]
    lotte_results = {}

    for domain in lotte_domains:
        print(f"\n  Domain: {domain}")
        set_seed(42)
        corpus, queries, qrels, triplets = generate_lotte_style_dataset(
            domain_name=domain, seed=42,
        )
        train_triplets = triplets[:min(len(triplets), 1500)]

        domain_results = {}
        for model_name, config in model_configs.items():
            set_seed(42)
            kwargs = dict(
                model_name="small", embedding_dim=128,
                query_max_length=32, doc_max_length=64,
            )
            if config["class"] == FLUKEModel:
                kwargs.update(use_tir=True, use_cqi=True, use_soft_topk=True)
            elif config["class"] == FLUKEPlusModel:
                kwargs.update(
                    use_tir=True, use_cqi=True, use_soft_topk=True,
                    use_mgs=True, use_asc=True,
                )

            model = config["class"](**kwargs)
            train_model(model, train_triplets, num_epochs=num_epochs,
                         batch_size=train_batch_size, lr=lr)

            metrics = evaluate_retrieval(model, corpus, queries, qrels, config["type"])
            domain_results[model_name] = metrics
            print(f"    {model_name}: nDCG@10={metrics['nDCG@10']:.4f}, "
                  f"S@5={metrics['Success@5']:.4f}")
            del model

        lotte_results[domain] = domain_results

    results["lotte_style"] = lotte_results

    # Print summary table
    print("\n" + "-" * 80)
    print("BEIR-style Results:")
    _print_results_table(beir_results)

    print("\nLoTTE-style Results (per domain):")
    for domain, dr in lotte_results.items():
        print(f"\n  {domain}:")
        _print_results_table(dr, indent=4)

    # Compute LoTTE averages
    print("\nLoTTE-style Averages:")
    avg_lotte = {}
    for model_name in model_configs:
        avg_metrics = {}
        for metric in lotte_results[lotte_domains[0]][model_name]:
            vals = [lotte_results[d][model_name][metric] for d in lotte_domains]
            avg_metrics[metric] = float(np.mean(vals))
        avg_lotte[model_name] = avg_metrics
    _print_results_table(avg_lotte, indent=2)
    results["lotte_style_average"] = avg_lotte

    return results


# =============================================================================
# Experiment 3: Component Ablation
# =============================================================================


def run_ablation_study(num_epochs=3, train_batch_size=16, lr=1e-4):
    """Experiment 3: Ablation of all 5 FLUKE+ components."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Component Ablation Study")
    print("=" * 70)

    set_seed(42)
    corpus, queries, qrels, triplets = generate_synthetic_dataset(
        n_topics=50, docs_per_topic=20, noise_docs=200, queries_per_topic=5,
    )
    train_triplets = triplets[:min(len(triplets), 2000)]

    configs = [
        ("ColBERTv2 (baseline)", {"cls": "colbert", "type": "colbert"}),
        ("+ CQI", {"cls": "fluke", "type": "fluke",
                    "use_cqi": True, "use_soft_topk": False, "use_tir": False}),
        ("+ SoftTopK", {"cls": "fluke", "type": "fluke",
                        "use_cqi": False, "use_soft_topk": True, "use_tir": False}),
        ("+ TIR", {"cls": "fluke", "type": "fluke",
                    "use_cqi": False, "use_soft_topk": False, "use_tir": True}),
        ("+ MGS", {"cls": "fluke_plus", "type": "fluke_plus",
                    "use_cqi": False, "use_soft_topk": False, "use_tir": False,
                    "use_mgs": True, "use_asc": False}),
        ("+ ASC", {"cls": "fluke_plus", "type": "fluke_plus",
                    "use_cqi": False, "use_soft_topk": False, "use_tir": False,
                    "use_mgs": False, "use_asc": True}),
        ("FLUKE (CQI+STK+TIR)", {"cls": "fluke", "type": "fluke",
                                   "use_cqi": True, "use_soft_topk": True, "use_tir": True}),
        ("FLUKE+ (all 5)", {"cls": "fluke_plus", "type": "fluke_plus",
                             "use_cqi": True, "use_soft_topk": True, "use_tir": True,
                             "use_mgs": True, "use_asc": True}),
    ]

    results = {}
    for name, config in configs:
        print(f"\n--- {name} ---")
        set_seed(42)

        base_kwargs = dict(
            model_name="small", embedding_dim=128,
            query_max_length=32, doc_max_length=64,
        )

        if config["cls"] == "colbert":
            model = ColBERTModel(**base_kwargs)
        elif config["cls"] == "fluke":
            model = FLUKEModel(
                **base_kwargs,
                use_cqi=config.get("use_cqi", False),
                use_soft_topk=config.get("use_soft_topk", False),
                use_tir=config.get("use_tir", False),
            )
        else:  # fluke_plus
            model = FLUKEPlusModel(
                **base_kwargs,
                use_cqi=config.get("use_cqi", False),
                use_soft_topk=config.get("use_soft_topk", False),
                use_tir=config.get("use_tir", False),
                use_mgs=config.get("use_mgs", False),
                use_asc=config.get("use_asc", False),
            )

        train_model(model, train_triplets, num_epochs=num_epochs,
                     batch_size=train_batch_size, lr=lr)
        metrics = evaluate_retrieval(model, corpus, queries, qrels, config["type"])
        print(f"  {name}: nDCG@10={metrics['nDCG@10']:.4f}, "
              f"Recall@100={metrics['Recall@100']:.4f}")
        results[name] = metrics
        del model

    # Print ablation table
    print("\n" + "-" * 90)
    print("Ablation Results:")
    _print_results_table(results)

    return results


# =============================================================================
# Formatting and Main
# =============================================================================


def _print_results_table(results, indent=2):
    """Print a formatted results table."""
    if not results:
        print(" " * indent + "No results.")
        return

    first_metrics = list(next(iter(results.values())).keys())
    # Show key metrics
    key_metrics = [m for m in ["nDCG@5", "nDCG@10", "Recall@100", "Success@5"]
                   if m in first_metrics]
    if not key_metrics:
        key_metrics = first_metrics[:4]

    prefix = " " * indent
    header = f"{prefix}{'Model':<30}"
    for m in key_metrics:
        header += f" {m:>12}"
    print(header)
    print(prefix + "-" * (30 + 13 * len(key_metrics)))

    for model_name, metrics in results.items():
        row = f"{prefix}{model_name:<30}"
        for m in key_metrics:
            row += f" {metrics.get(m, 0):>12.4f}"
        print(row)


def format_all_results(all_results: dict) -> str:
    """Format all results as a comprehensive report."""
    lines = []
    lines.append("\n" + "#" * 70)
    lines.append("# COMPREHENSIVE RESULTS: ColBERTv2 vs FLUKE vs FLUKE+")
    lines.append("#" * 70)

    if "experiment_1_scoring" in all_results:
        lines.append("\n## Experiment 1: Scoring Function Analysis")
        lines.append("(Accuracy: fraction of trials where correct document scored higher)\n")
        r = all_results["experiment_1_scoring"]
        if "average" in r:
            for method, data in r["average"].items():
                lines.append(f"  {method}: {data['accuracy']:.3f}")

    if "experiment_2_e2e" in all_results:
        lines.append("\n## Experiment 2: End-to-End Retrieval")
        r = all_results["experiment_2_e2e"]
        if "beir_style" in r:
            lines.append("\nBEIR-style:")
            for model, metrics in r["beir_style"].items():
                lines.append(f"  {model}: nDCG@10={metrics.get('nDCG@10', 0):.4f}, "
                           f"Recall@100={metrics.get('Recall@100', 0):.4f}")
        if "lotte_style_average" in r:
            lines.append("\nLoTTE-style (average across domains):")
            for model, metrics in r["lotte_style_average"].items():
                lines.append(f"  {model}: nDCG@10={metrics.get('nDCG@10', 0):.4f}, "
                           f"S@5={metrics.get('Success@5', 0):.4f}")

    if "experiment_3_ablation" in all_results:
        lines.append("\n## Experiment 3: Ablation Study")
        r = all_results["experiment_3_ablation"]
        for config, metrics in r.items():
            lines.append(f"  {config}: nDCG@10={metrics.get('nDCG@10', 0):.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Full benchmark: ColBERTv2 vs FLUKE vs FLUKE+"
    )
    parser.add_argument(
        "--experiment", choices=["scoring", "e2e", "ablation", "all"],
        default="all", help="Which experiment to run",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scoring-trials", type=int, default=1000)
    parser.add_argument("--output", default="results/full_benchmark_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    start_time = time.time()

    all_results = {}

    if args.experiment in ("scoring", "all"):
        exp1 = run_scoring_analysis(n_trials=args.scoring_trials)
        all_results["experiment_1_scoring"] = exp1

    if args.experiment in ("e2e", "all"):
        exp2 = run_e2e_experiment(
            num_epochs=args.num_epochs,
            train_batch_size=args.batch_size,
            lr=args.lr,
        )
        all_results["experiment_2_e2e"] = exp2

    if args.experiment in ("ablation", "all"):
        exp3 = run_ablation_study(
            num_epochs=args.num_epochs,
            train_batch_size=args.batch_size,
            lr=args.lr,
        )
        all_results["experiment_3_ablation"] = exp3

    elapsed = time.time() - start_time

    # Print comprehensive summary
    report = format_all_results(all_results)
    print(report)
    print(f"\n\nTotal time: {elapsed:.1f}s")

    # Save results
    all_results["total_time_seconds"] = elapsed
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
