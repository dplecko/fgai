# Causal Bias Detection for Generative AI

This works studies the foundations of causal bias detection for generative artificial intelligence (AI) models.

---

## Part I — LLM Data Generation (`py/`)

The `python` pipeline elicits the observational distributions of language models by prompting
them to write stories based on with different conditioning sets (∅ → {X, Z} → {X, Z, W}, where X, Z, W corresponding to variable in standard fairness model).

### File Organization

| File | Purpose |
|---|---|
| `py/elicit.py` | Main entry point: samples rows from real data, runs LLM generation and annotation steps |
| `py/generation.py` | Builds supporting functionality for `py/elicit.py`; supports `transformers` and `vllm` backends |
| `py/data_helpers.py` | Helpers for data loading |
| `py/model_load.py` | Modeling loading/specification |

---

## Part 2 — Causal Estimation (`r/`)

The `R` side loads the partially-generated datasets for each (dataset, model) pair and
estimates the causal fairness measures using a one-step debiasing.

### File Organization

| File | Purpose |
|---|---|
| `scripts/waterfall.R` | Runs `estimate_within` for one (dataset, model) and saves a TV decomposition waterfall plot |
| `scripts/stereo.R` | Summarizes  |
| `scripts/similarity.R` | Computes bias signature similarity between models |
| `scripts/families.R` | Analyzes if models from the same family have similar bias signatures |
| `r/helpers.R` | Key functionality for data analysis |
| `r/one-step-debias.R` | Core causal fairness estimation |

