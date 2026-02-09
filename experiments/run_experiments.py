#!/usr/bin/env python3
"""Comprehensive experiments comparing FLUKE vs ColBERTv2.

Three experiment suites:

1. Scoring Function Analysis (Experiment 1):
   Direct comparison of scoring functions using synthetic embeddings.
   Tests specific weaknesses of MaxSim that FLUKE addresses:
   - Query term importance sensitivity
   - Robustness to spurious matches
   - Cross-term dependency modeling

2. End-to-End Retrieval (Experiment 2):
   Train both models from scratch on synthetic IR data, evaluate retrieval.

3. Ablation Study (Experiment 3):
   Isolate contribution of each FLUKE component (CQI, SoftTopK, TIR).

Usage:
    python experiments/run_experiments.py
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

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
from fluke.models.colbert import ColBERTModel
from fluke.models.fluke_model import FLUKEModel
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


def generate_controlled_embeddings(dim=128, seed=42):
    """Generate controlled embeddings to test specific scoring weaknesses.

    Creates scenarios where ColBERTv2's MaxSim fails but FLUKE succeeds.
    """
    torch.manual_seed(seed)

    scenarios = {}

    # Scenario 1: Query Term Importance
    # Query: [important_term, filler_term1, filler_term2]
    # Doc A: matches important_term well, filler poorly -> should be relevant
    # Doc B: matches fillers well, important poorly -> should be irrelevant
    # MaxSim treats them equally; CQI should differentiate
    important = F.normalize(torch.randn(1, dim), dim=-1)
    filler1 = F.normalize(torch.randn(1, dim), dim=-1)
    filler2 = F.normalize(torch.randn(1, dim), dim=-1)

    query = torch.cat([important, filler1, filler2], dim=0)

    # Doc A: strong match to important, weak to fillers
    doc_a = torch.cat([
        F.normalize(important + 0.05 * torch.randn(1, dim), dim=-1),
        F.normalize(torch.randn(1, dim), dim=-1),  # random
        F.normalize(torch.randn(1, dim), dim=-1),
        F.normalize(torch.randn(1, dim), dim=-1),
    ], dim=0)

    # Doc B: strong match to fillers, weak to important
    doc_b = torch.cat([
        F.normalize(torch.randn(1, dim), dim=-1),  # random
        F.normalize(filler1 + 0.05 * torch.randn(1, dim), dim=-1),
        F.normalize(filler2 + 0.05 * torch.randn(1, dim), dim=-1),
        F.normalize(torch.randn(1, dim), dim=-1),
    ], dim=0)

    scenarios["term_importance"] = {
        "query": query,
        "relevant_doc": doc_a,
        "irrelevant_doc": doc_b,
        "description": "Important query term vs filler terms",
    }

    # Scenario 2: Spurious High-Similarity Match
    # One doc token happens to have high similarity to a query token by chance,
    # but the rest of the document is irrelevant
    real_topic = F.normalize(torch.randn(1, dim), dim=-1)
    query_tokens = torch.cat([
        real_topic,
        F.normalize(real_topic + 0.3 * torch.randn(1, dim), dim=-1),
        F.normalize(real_topic + 0.3 * torch.randn(1, dim), dim=-1),
    ], dim=0)

    # Doc C: genuinely about the topic (moderate matches to all query tokens)
    doc_c = torch.cat([
        F.normalize(real_topic + 0.2 * torch.randn(1, dim), dim=-1),
        F.normalize(real_topic + 0.25 * torch.randn(1, dim), dim=-1),
        F.normalize(real_topic + 0.3 * torch.randn(1, dim), dim=-1),
        F.normalize(torch.randn(1, dim), dim=-1),
    ], dim=0)

    # Doc D: spurious match - one token extremely similar to query token 1,
    # but rest is random noise
    doc_d = torch.cat([
        F.normalize(real_topic + 0.01 * torch.randn(1, dim), dim=-1),  # near-perfect match
        F.normalize(torch.randn(1, dim), dim=-1),  # random
        F.normalize(torch.randn(1, dim), dim=-1),
        F.normalize(torch.randn(1, dim), dim=-1),
    ], dim=0)

    scenarios["spurious_match"] = {
        "query": query_tokens,
        "relevant_doc": doc_c,
        "irrelevant_doc": doc_d,
        "description": "Genuinely relevant doc vs spurious single-token match",
    }

    # Scenario 3: Cross-term Dependencies (Conjunction)
    # Query has two terms that BOTH need to match
    # Doc E: matches both terms -> relevant
    # Doc F: matches only one term very well -> irrelevant
    term_a = F.normalize(torch.randn(1, dim), dim=-1)
    term_b = F.normalize(torch.randn(1, dim), dim=-1)
    query_conj = torch.cat([term_a, term_b], dim=0)

    # Doc E: matches both
    doc_e = torch.cat([
        F.normalize(term_a + 0.1 * torch.randn(1, dim), dim=-1),
        F.normalize(term_b + 0.1 * torch.randn(1, dim), dim=-1),
        F.normalize(torch.randn(1, dim), dim=-1),
    ], dim=0)

    # Doc F: matches only term_a very well, term_b poorly
    doc_f = torch.cat([
        F.normalize(term_a + 0.02 * torch.randn(1, dim), dim=-1),  # excellent match
        F.normalize(term_a + 0.02 * torch.randn(1, dim), dim=-1),  # another match for a
        F.normalize(torch.randn(1, dim), dim=-1),  # no match for b
    ], dim=0)

    scenarios["conjunction"] = {
        "query": query_conj,
        "relevant_doc": doc_e,
        "irrelevant_doc": doc_f,
        "description": "Both query terms must match (conjunction)",
    }

    return scenarios


def run_scoring_analysis():
    """Experiment 1: Direct scoring function comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Scoring Function Analysis")
    print("=" * 70)
    print("Testing specific scenarios where FLUKE's innovations should help\n")

    dim = 128
    scenarios = generate_controlled_embeddings(dim=dim)

    # Initialize TIR module (randomly, since this is a controlled test)
    tir = TokenInteractionResidual(max_query_tokens=32, hidden_dim=64)

    results = {}

    for scenario_name, data in scenarios.items():
        print(f"\n--- Scenario: {data['description']} ---")
        query = data["query"]
        rel_doc = data["relevant_doc"]
        irr_doc = data["irrelevant_doc"]

        # ColBERTv2 MaxSim
        maxsim_rel = maxsim(query, rel_doc).item()
        maxsim_irr = maxsim(query, irr_doc).item()
        maxsim_correct = maxsim_rel > maxsim_irr

        # FLUKE with uniform weights (ablation: just SoftTopK)
        uniform_weights = torch.ones(query.shape[0])
        fluke_stk_rel = fluke_score(query, rel_doc, uniform_weights, topk=3, temperature=0.1, max_query_tokens=32).item()
        fluke_stk_irr = fluke_score(query, irr_doc, uniform_weights, topk=3, temperature=0.1, max_query_tokens=32).item()
        stk_correct = fluke_stk_rel > fluke_stk_irr

        # FLUKE with importance weights (ablation: CQI effect)
        # Simulate CQI: give higher weight to first query token (the "important" one)
        if scenario_name == "term_importance":
            cqi_weights = torch.tensor([3.0, 1.0, 1.0])
        elif scenario_name == "conjunction":
            cqi_weights = torch.tensor([1.5, 1.5])
        else:
            cqi_weights = torch.ones(query.shape[0])

        fluke_cqi_rel = fluke_score(query, rel_doc, cqi_weights, topk=3, temperature=0.1, max_query_tokens=32).item()
        fluke_cqi_irr = fluke_score(query, irr_doc, cqi_weights, topk=3, temperature=0.1, max_query_tokens=32).item()
        cqi_correct = fluke_cqi_rel > fluke_cqi_irr

        print(f"  MaxSim  — Relevant: {maxsim_rel:.4f}, Irrelevant: {maxsim_irr:.4f}, "
              f"Correct: {maxsim_correct}")
        print(f"  SoftTopK — Relevant: {fluke_stk_rel:.4f}, Irrelevant: {fluke_stk_irr:.4f}, "
              f"Correct: {stk_correct}")
        print(f"  CQI+STK — Relevant: {fluke_cqi_rel:.4f}, Irrelevant: {fluke_cqi_irr:.4f}, "
              f"Correct: {cqi_correct}")

        results[scenario_name] = {
            "maxsim": {"relevant": maxsim_rel, "irrelevant": maxsim_irr, "correct": maxsim_correct},
            "soft_topk": {"relevant": fluke_stk_rel, "irrelevant": fluke_stk_irr, "correct": stk_correct},
            "cqi_stk": {"relevant": fluke_cqi_rel, "irrelevant": fluke_cqi_irr, "correct": cqi_correct},
        }

    # Run statistical analysis with many random instances
    print("\n\n--- Statistical Analysis (1000 random instances per scenario) ---\n")
    stat_results = run_statistical_scoring_analysis(dim=dim, n_trials=1000)
    results["statistical"] = stat_results

    return results


def run_statistical_scoring_analysis(dim=128, n_trials=1000):
    """Run many random instances of each scenario to get statistical significance."""
    set_seed(42)

    scenario_types = ["term_importance", "spurious_match", "conjunction"]
    methods = ["maxsim", "soft_topk", "cqi_stk"]
    accuracies = {s: {m: 0 for m in methods} for s in scenario_types}
    margins = {s: {m: [] for m in methods} for s in scenario_types}

    for trial in range(n_trials):
        # Term Importance scenario
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

        ms_r = maxsim(query, doc_rel).item()
        ms_i = maxsim(query, doc_irr).item()
        accuracies["term_importance"]["maxsim"] += int(ms_r > ms_i)
        margins["term_importance"]["maxsim"].append(ms_r - ms_i)

        uw = torch.ones(3)
        stk_r = fluke_score(query, doc_rel, uw, topk=3, max_query_tokens=32).item()
        stk_i = fluke_score(query, doc_irr, uw, topk=3, max_query_tokens=32).item()
        accuracies["term_importance"]["soft_topk"] += int(stk_r > stk_i)
        margins["term_importance"]["soft_topk"].append(stk_r - stk_i)

        cqi_w = torch.tensor([3.0, 1.0, 1.0])
        cqi_r = fluke_score(query, doc_rel, cqi_w, topk=3, max_query_tokens=32).item()
        cqi_i = fluke_score(query, doc_irr, cqi_w, topk=3, max_query_tokens=32).item()
        accuracies["term_importance"]["cqi_stk"] += int(cqi_r > cqi_i)
        margins["term_importance"]["cqi_stk"].append(cqi_r - cqi_i)

        # Spurious Match scenario
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

        uw3 = torch.ones(3)
        for method in methods:
            if method == "maxsim":
                r = maxsim(q_sp, doc_genuine).item()
                i = maxsim(q_sp, doc_spurious).item()
            elif method == "soft_topk":
                r = fluke_score(q_sp, doc_genuine, uw3, topk=3, max_query_tokens=32).item()
                i = fluke_score(q_sp, doc_spurious, uw3, topk=3, max_query_tokens=32).item()
            else:
                r = fluke_score(q_sp, doc_genuine, uw3, topk=3, max_query_tokens=32).item()
                i = fluke_score(q_sp, doc_spurious, uw3, topk=3, max_query_tokens=32).item()
            accuracies["spurious_match"][method] += int(r > i)
            margins["spurious_match"][method].append(r - i)

        # Conjunction scenario
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

        uw2 = torch.ones(2)
        cqi_conj = torch.tensor([1.5, 1.5])
        for method in methods:
            if method == "maxsim":
                r = maxsim(q_conj, doc_both).item()
                i = maxsim(q_conj, doc_one).item()
            elif method == "soft_topk":
                r = fluke_score(q_conj, doc_both, uw2, topk=3, max_query_tokens=32).item()
                i = fluke_score(q_conj, doc_one, uw2, topk=3, max_query_tokens=32).item()
            else:
                r = fluke_score(q_conj, doc_both, cqi_conj, topk=3, max_query_tokens=32).item()
                i = fluke_score(q_conj, doc_one, cqi_conj, topk=3, max_query_tokens=32).item()
            accuracies["conjunction"][method] += int(r > i)
            margins["conjunction"][method].append(r - i)

    # Print results
    stat_results = {}
    for scenario in scenario_types:
        print(f"\n  {scenario}:")
        stat_results[scenario] = {}
        for method in methods:
            acc = accuracies[scenario][method] / n_trials
            avg_margin = np.mean(margins[scenario][method])
            std_margin = np.std(margins[scenario][method])
            print(f"    {method:12s}: accuracy={acc:.3f}, margin={avg_margin:.4f} ± {std_margin:.4f}")
            stat_results[scenario][method] = {
                "accuracy": acc,
                "mean_margin": float(avg_margin),
                "std_margin": float(std_margin),
            }

    return stat_results


# =============================================================================
# Experiment 2: End-to-End Training + Retrieval
# =============================================================================


def generate_synthetic_ir_dataset(
    n_topics: int = 50,
    docs_per_topic: int = 20,
    noise_docs: int = 200,
    queries_per_topic: int = 5,
    words_per_doc: int = 30,
    words_per_query: int = 6,
    seed: int = 42,
):
    """Generate a synthetic IR dataset with controlled relevance.

    Each topic has a set of related "vocabulary" words. Documents about a topic
    use words from that topic's vocabulary. Queries are generated from topic words.
    Relevance is based on topic overlap.

    Returns:
        corpus: dict[doc_id -> text]
        queries: dict[query_id -> text]
        qrels: dict[query_id -> dict[doc_id -> relevance]]
        triplets: list of (query, pos_doc, neg_doc) for training
    """
    rng = random.Random(seed)

    # Generate topic vocabularies (each topic has ~20 unique-ish words)
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
        # Each topic picks 15-25 words (some overlap between topics is fine)
        n_words = rng.randint(15, 25)
        vocab = rng.sample(base_words, min(n_words, len(base_words)))
        # Add topic-specific prefix to make some words unique
        prefixed = [f"t{t}{w}" for w in rng.sample(vocab[:5], min(5, len(vocab)))]
        topic_vocabs.append(vocab + prefixed)

    # Generate documents
    corpus = {}
    doc_topics = {}  # doc_id -> topic_id
    doc_id_counter = 0

    for topic_id in range(n_topics):
        for _ in range(docs_per_topic):
            doc_id = f"doc_{doc_id_counter}"
            words = [rng.choice(topic_vocabs[topic_id]) for _ in range(words_per_doc)]
            # Add some noise words from other topics
            for _ in range(words_per_doc // 5):
                other_topic = rng.randint(0, n_topics - 1)
                words.append(rng.choice(topic_vocabs[other_topic]))
            rng.shuffle(words)
            corpus[doc_id] = " ".join(words)
            doc_topics[doc_id] = topic_id
            doc_id_counter += 1

    # Noise documents (not about any specific topic)
    for _ in range(noise_docs):
        doc_id = f"doc_{doc_id_counter}"
        words = [rng.choice(base_words) for _ in range(words_per_doc)]
        corpus[doc_id] = " ".join(words)
        doc_topics[doc_id] = -1
        doc_id_counter += 1

    # Generate queries
    queries = {}
    qrels = defaultdict(dict)
    query_id_counter = 0

    for topic_id in range(n_topics):
        for _ in range(queries_per_topic):
            query_id = f"q_{query_id_counter}"
            words = [rng.choice(topic_vocabs[topic_id]) for _ in range(words_per_query)]
            queries[query_id] = " ".join(words)

            # Mark relevant documents
            for doc_id, doc_topic in doc_topics.items():
                if doc_topic == topic_id:
                    qrels[query_id][doc_id] = 1

            query_id_counter += 1

    # Generate training triplets
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
        for _ in range(3):  # 3 triplets per query
            pos_doc_id = rng.choice(pos_docs)
            neg_doc_id = rng.choice(all_doc_ids)
            while doc_topics.get(neg_doc_id) == topic_id:
                neg_doc_id = rng.choice(all_doc_ids)
            triplets.append((query_text, corpus[pos_doc_id], corpus[neg_doc_id]))

    rng.shuffle(triplets)

    print(f"Generated synthetic IR dataset:")
    print(f"  Topics: {n_topics}")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {sum(len(v) for v in qrels.values())} judgments")
    print(f"  Training triplets: {len(triplets)}")

    return corpus, queries, dict(qrels), triplets


def evaluate_retrieval(
    model, corpus, queries, qrels, model_type="colbert",
    batch_size=64, top_k=100,
):
    """Evaluate a model on retrieval task."""
    import pytrec_eval

    model.eval()

    # Encode corpus
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_embs = model.encode_documents(doc_texts, batch_size=batch_size, show_progress=True)

    # Build index
    index = TokenEmbeddingIndex()
    index.add_batch(doc_ids, doc_embs)

    # Encode queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    if model_type == "fluke":
        query_data = model.encode_queries(query_texts, batch_size=batch_size)
    else:
        query_data = model.encode_queries(query_texts, batch_size=batch_size)

    # Search
    tir_module = getattr(model, "tir", None)
    searcher = LatentSearcher(
        index, scoring=model_type, tir_module=tir_module,
        max_query_tokens=getattr(model, "query_max_length", 32),
    )
    search_results = searcher.batch_search(query_data, top_k=top_k, show_progress=True)

    # Format results
    formatted = {}
    for q_idx, results_list in search_results.items():
        qid = query_ids[q_idx]
        formatted[qid] = {doc_id: score for doc_id, score in results_list}

    # Compute metrics
    for k in [10, 100]:
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {f"ndcg_cut_{k}", f"map_cut_{k}", f"recall_{k}"}
        )
        scores = evaluator.evaluate(formatted)

        ndcg = np.mean([s[f"ndcg_cut_{k}"] for s in scores.values()])
        map_score = np.mean([s[f"map_cut_{k}"] for s in scores.values()])
        recall = np.mean([s[f"recall_{k}"] for s in scores.values()])

    # Return main metrics
    evaluator10 = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "map_cut_10", "recall_100"}
    )
    evaluator100 = pytrec_eval.RelevanceEvaluator(
        qrels, {"recall_100"}
    )
    scores10 = evaluator10.evaluate(formatted)
    scores100 = evaluator100.evaluate(formatted)

    metrics = {
        "nDCG@10": float(np.mean([s["ndcg_cut_10"] for s in scores10.values()])),
        "MAP@10": float(np.mean([s["map_cut_10"] for s in scores10.values()])),
        "Recall@100": float(np.mean([s["recall_100"] for s in scores100.values()])),
    }

    return metrics


def run_end_to_end_experiment(
    n_topics=50, num_epochs=3, train_batch_size=16, lr=1e-4,
):
    """Experiment 2: End-to-end training and retrieval comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: End-to-End Training + Retrieval")
    print("=" * 70)

    set_seed(42)

    # Generate dataset
    corpus, queries, qrels, triplets = generate_synthetic_ir_dataset(
        n_topics=n_topics, docs_per_topic=20, noise_docs=200,
        queries_per_topic=5,
    )

    # Use only first N triplets for training
    train_triplets = triplets[:min(len(triplets), 2000)]

    results = {}

    # --- ColBERTv2 Baseline ---
    print("\n--- Training ColBERTv2 ---")
    colbert = ColBERTModel(model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64)
    train_model(
        colbert, train_triplets, num_epochs=num_epochs,
        batch_size=train_batch_size, lr=lr,
    )

    print("\n--- Evaluating ColBERTv2 ---")
    colbert_metrics = evaluate_retrieval(colbert, corpus, queries, qrels, "colbert")
    print(f"ColBERTv2: {colbert_metrics}")
    results["ColBERTv2"] = colbert_metrics

    # Get ColBERTv2 encoder state for fair comparison
    colbert_state = colbert.encoder.state_dict()

    # --- FLUKE (all components) ---
    print("\n--- Training FLUKE (CQI + SoftTopK + TIR) ---")
    fluke = FLUKEModel(
        model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64,
        use_tir=True, use_cqi=True, use_soft_topk=True,
    )
    # Initialize encoder from same weights as ColBERTv2 for fair comparison
    # (Both start from identical encoder, only scoring differs)
    # Actually train from same random init by using same seed
    set_seed(42)
    fluke = FLUKEModel(
        model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64,
        use_tir=True, use_cqi=True, use_soft_topk=True,
    )
    train_model(
        fluke, train_triplets, num_epochs=num_epochs,
        batch_size=train_batch_size, lr=lr,
    )

    print("\n--- Evaluating FLUKE ---")
    fluke_metrics = evaluate_retrieval(fluke, corpus, queries, qrels, "fluke")
    print(f"FLUKE: {fluke_metrics}")
    results["FLUKE"] = fluke_metrics

    return results


# =============================================================================
# Experiment 3: Ablation Study
# =============================================================================


def run_ablation_study(
    n_topics=50, num_epochs=3, train_batch_size=16, lr=1e-4,
):
    """Experiment 3: Ablation study on FLUKE components."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Ablation Study")
    print("=" * 70)

    # Generate dataset (same as experiment 2 for consistency)
    set_seed(42)
    corpus, queries, qrels, triplets = generate_synthetic_ir_dataset(
        n_topics=n_topics, docs_per_topic=20, noise_docs=200,
        queries_per_topic=5,
    )
    train_triplets = triplets[:min(len(triplets), 2000)]

    configs = [
        ("ColBERTv2 (baseline)", {"model_class": "colbert"}),
        ("+ CQI only", {"use_cqi": True, "use_soft_topk": False, "use_tir": False}),
        ("+ SoftTopK only", {"use_cqi": False, "use_soft_topk": True, "use_tir": False}),
        ("+ TIR only", {"use_cqi": False, "use_soft_topk": False, "use_tir": True}),
        ("+ CQI + SoftTopK", {"use_cqi": True, "use_soft_topk": True, "use_tir": False}),
        ("+ CQI + TIR", {"use_cqi": True, "use_soft_topk": False, "use_tir": True}),
        ("FLUKE (all)", {"use_cqi": True, "use_soft_topk": True, "use_tir": True}),
    ]

    results = {}
    for name, config in configs:
        print(f"\n--- {name} ---")
        set_seed(42)  # Same init for all

        if config.get("model_class") == "colbert":
            model = ColBERTModel(model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64)
            model_type = "colbert"
        else:
            model = FLUKEModel(
                model_name="small", embedding_dim=128, query_max_length=32, doc_max_length=64,
                **{k: v for k, v in config.items() if k != "model_class"},
            )
            model_type = "fluke"

        train_model(model, train_triplets, num_epochs=num_epochs, batch_size=train_batch_size, lr=lr)
        metrics = evaluate_retrieval(model, corpus, queries, qrels, model_type)
        print(f"  {name}: {metrics}")
        results[name] = metrics
        del model

    return results


# =============================================================================
# Results Formatting
# =============================================================================


def format_results_table(results: dict, title: str) -> str:
    """Format results as a readable ASCII table."""
    lines = [f"\n{title}", "=" * 80]

    if not results:
        return "\n".join(lines + ["No results."])

    first_val = next(iter(results.values()))
    if isinstance(first_val, dict) and all(isinstance(v, (int, float)) for v in first_val.values()):
        # Flat: {config -> {metric -> value}}
        metrics = sorted(first_val.keys())
        header = f"{'Configuration':<30}"
        for m in metrics:
            header += f" {m:>12}"
        lines.append(header)
        lines.append("-" * 80)
        for config_name, config_metrics in results.items():
            row = f"{config_name:<30}"
            for m in metrics:
                row += f" {config_metrics.get(m, 0):>12.4f}"
            lines.append(row)
    else:
        lines.append(json.dumps(results, indent=2))

    return "\n".join(lines)


def main():
    os.makedirs("results", exist_ok=True)
    set_seed(42)
    start_time = time.time()

    all_results = {}

    # Experiment 1: Scoring Analysis
    print("\n" + "#" * 70)
    print("# Running all experiments")
    print("#" * 70)

    exp1_results = run_scoring_analysis()
    all_results["experiment_1_scoring"] = exp1_results

    # Experiment 2: End-to-End
    exp2_results = run_end_to_end_experiment(
        n_topics=50, num_epochs=3, train_batch_size=16, lr=1e-4,
    )
    all_results["experiment_2_e2e"] = exp2_results

    # Experiment 3: Ablation
    exp3_results = run_ablation_study(
        n_topics=50, num_epochs=3, train_batch_size=16, lr=1e-4,
    )
    all_results["experiment_3_ablation"] = exp3_results

    elapsed = time.time() - start_time

    # Print summary
    print("\n\n" + "#" * 70)
    print("# RESULTS SUMMARY")
    print("#" * 70)

    print(format_results_table(exp2_results, "Experiment 2: End-to-End Retrieval"))
    print(format_results_table(exp3_results, "Experiment 3: Ablation Study"))

    print(f"\n\nTotal experiment time: {elapsed:.1f}s")

    # Save results
    all_results["total_time_seconds"] = elapsed
    output_path = "results/experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
