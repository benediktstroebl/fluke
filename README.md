# FLUKE: Fused Late-interaction with Unified Key Embeddings

A novel neural information retrieval method that improves upon ColBERTv2's late interaction paradigm through three key innovations.

## Method

### Background: ColBERTv2 Late Interaction

ColBERTv2 scores query-document relevance using **MaxSim**: encode query and document independently into per-token embeddings, then for each query token find its maximum cosine similarity to any document token, and sum:

```
S_colbert(q, d) = Σ_i  max_j  (q_i · d_j)
```

This treats all query tokens equally and considers each independently.

### FLUKE Innovations

FLUKE introduces three targeted improvements while **preserving ColBERTv2's indexing efficiency** (documents are still encoded offline into per-token embeddings):

#### 1. Contextual Query Importance (CQI)

Not all query tokens are equally informative. "photosynthesis" carries more signal than "the" in a relevance judgment. CQI learns per-token importance weights from query context:

```
S_cqi(q, d) = Σ_i  w(q_i) · max_j (q_i · d_j)
```

where `w(q_i)` is derived from a lightweight attention mechanism using the [CLS] token's representation attending over all query tokens. Weights are softmax-normalized and scaled to preserve score magnitude.

#### 2. Soft Top-K Aggregation

ColBERTv2's hard `max` over document tokens is susceptible to spurious high-similarity matches (a single noisy token can dominate). FLUKE replaces this with a **softmax-weighted aggregation** of the top-K most similar document tokens:

```
SoftTopK(q_i, D) = Σ_{j ∈ TopK(q_i, D)}  softmax(q_i · d_j / τ) · (q_i · d_j)
```

This smooths out noise while still focusing on the most relevant matches. The temperature parameter τ controls sharpness (τ→0 recovers hard max).

#### 3. Token Interaction Residual (TIR)

MaxSim treats each query token's match independently — it cannot capture **cross-term dependencies**. For example, a query "python machine learning library" should require all three concepts to co-occur meaningfully in the document. TIR adds a lightweight MLP that operates on the vector of per-token match scores:

```
S_fluke(q, d) = S_cqi(q, d) + MLP([score_1, score_2, ..., score_|q|])
```

The MLP input is only |q|-dimensional (typically 32), making it extremely cheap. Yet it can learn patterns like:
- **Conjunction**: "both terms A and B must match well"
- **Required terms**: "term A matching without B indicates irrelevance"
- **Score calibration**: non-linear corrections to the base score

The MLP is initialized near-zero so FLUKE starts identical to ColBERTv2 and the residual is learned as a refinement.

### Why It Works

| Component | What It Fixes | Intuition |
|-----------|--------------|-----------|
| CQI | Uniform query token weighting | Rare/specific terms should dominate scoring |
| Soft Top-K | Brittle hard maximum | Robustness to noise, captures multi-context matches |
| TIR | Independent per-token scoring | Captures term co-occurrence patterns |

### Efficiency

- **Document encoding**: Identical to ColBERTv2 (offline, per-token embeddings)
- **Query encoding**: CQI adds one attention head (~0.5% overhead)
- **Scoring**: Soft Top-K and TIR add negligible compute (operates on 32-dim vectors)
- **Storage**: Same as ColBERTv2 (per-token document embeddings)

## Project Structure

```
fluke/
├── fluke/
│   ├── models/
│   │   ├── colbert.py          # ColBERTv2 baseline
│   │   ├── fluke_model.py      # FLUKE model
│   │   └── encoders.py         # Shared BERT token encoder
│   ├── scoring/
│   │   ├── maxsim.py           # Standard MaxSim scoring
│   │   └── fluke_scoring.py    # CQI + SoftTopK + TIR scoring
│   ├── indexing/
│   │   ├── indexer.py          # Token embedding index
│   │   └── searcher.py         # Brute-force search
│   ├── training/
│   │   └── trainer.py          # Contrastive training loop
│   └── evaluation/
│       └── benchmarks.py       # BEIR benchmark evaluation
├── experiments/
│   └── run_benchmark.py        # Main experiment runner
└── results/                    # Benchmark results
```

## Experiments

### Datasets

We evaluate on [BEIR](https://github.com/beir-cellar/beir) benchmarks:
- **SciFact**: Scientific claim verification (5K docs, 300 queries)
- **NFCorpus**: Nutrition/medical retrieval (3.6K docs, 323 queries)

### Metrics

- **nDCG@10**: Normalized discounted cumulative gain at rank 10
- **MAP@10**: Mean average precision at rank 10
- **Recall@100**: Fraction of relevant documents retrieved in top 100

### Running Experiments

```bash
# Train on MS MARCO, evaluate on BEIR
python experiments/run_benchmark.py --mode train_eval --datasets scifact nfcorpus

# Zero-shot evaluation (no training)
python experiments/run_benchmark.py --mode zero_shot --datasets scifact nfcorpus

# Ablation study (component-by-component)
python experiments/run_benchmark.py --mode ablation --datasets scifact
```

## Installation

```bash
pip install torch transformers datasets pytrec-eval-terrier scikit-learn tqdm
```

## Citation

If you use FLUKE in your research, please cite this repository.
