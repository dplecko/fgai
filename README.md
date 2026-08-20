# Causal Bias Detection for Generative AI

This works studies the foundations of causal bias detection for generative artificial intelligence (AI) models.

---

## Part I: LLM Data Generation (`py/`)

The `python` pipeline elicits the observational distributions of language models by prompting
them to write stories based on with different conditioning sets (∅ → {X, Z} → {X, Z, W}, where X, Z, W correspond to variable sets in the standard fairness model).

### File Organization

| File | Purpose |
|---|---|
| `py/elicit.py` | Main entry point: samples rows from real data, runs LLM generation and annotation steps for the required variables |
| `py/generation.py` | Builds supporting functionality for `py/elicit.py` |
| `py/data_helpers.py` | Helpers for data loading |
| `py/model_load.py` | Modeling loading and specification |

---

## Part 2: Causal Estimation (`r/`)

The `R` side loads the (partially) generated datasets for each (dataset, model) pair and
estimates the causal fairness measures using one-step debiasing.

### File Organization

The key scripts are the following:

| File | Purpose |
|---|---|
| `scripts/bias-stereo.R` | Summarizes the overall model behavior across datasets and stages |
| `scripts/bias-similarity.R` | Computes bias signature similarity between models |
| `scripts/bias-families.R` | Analyzes if models from the same family have similar bias signatures |
| `scripts/case-studies-select.R` | Selects interesting case-studies |
| `scripts/case-studies-wfall.R` | Generates waterfall plots for the selected case studies |
| `scripts/annotator-agreement.R` | Cross-annotator agreement (Llama 3 70B vs Command A 111B) |
| `scripts/annotator-vs-human.R` | Compares annotator output against human-labeled ground truth |
| `scripts/annotator-sensitivity.R` | Sensitivity of estimates to annotator choice |
| `scripts/annotator-na-analysis.R` | Analyzes "Answer not available" rates across annotation attempts |
| `scripts/sens-prompt.R` | Sensitivity to prompt format (narrative vs. bulleted list) |
| `scripts/sens-tempp.R` | Sensitivity to decoding parameters (temperature / top-p) |

Furthermore, the `r/` infrastructure is organized as:

| File | Purpose |
|---|---|
| `r/helpers.R` | Shared core: `load_model_data` → `estimate_within` → `extract_stage_effects` |
| `r/one-step-debias.R` | Core causal fairness estimation (one-step debiasing) |
| `r/helpers-stereo.R` | Scoring/classification helpers (amplify / dampen / reverse / no bias) |
| `r/helpers-similarity.R` | 9D bias-vector extraction for similarity comparisons |
| `r/helpers-wfall.R` | Waterfall (TV decomposition) plotting helpers |
| `r/zzz-deps.R` | Package dependencies |