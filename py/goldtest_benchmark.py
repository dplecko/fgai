# Score an annotator model against the 300-row Claude-labeled gold test set
# (data/gold_test/gold_test_labeled.parquet), using the real generation.py
# annotate_data pipeline (vLLM engine) -- not a reimplementation.
#
# Usage (run inside the fgai container, needs a GPU):
#   python3 -m py.goldtest_benchmark --ann_model qwen25_72b
#   python3 -m py.goldtest_benchmark --ann_model qwen25_72b --few_shot
#
# The 300 gold rows span many (dataset, variable) pairs, but annotate_data's
# API assumes one shared var_dict per call -- so this groups rows by
# (dataset, variable) and calls annotate_data once per group, exactly as
# elicit.py does per group_name, just re-keyed. Each group has exactly one
# "attempt" (the fixed gold story), no retries.
import argparse
import os

import pandas as pd

from py.model_load import MODEL_PATHS, get_vllm_model
from py.generation import annotate_data
from py.data_helpers import load_data

GOLD_LABELED_PATH = "data/gold_test/gold_test_labeled.parquet"

_dataset_info_cache = {}


def dataset_info(dataset):
    if dataset not in _dataset_info_cache:
        _df, var_dict, var_names, var_ord, _sfm, _context = load_data(dataset)
        _dataset_info_cache[dataset] = {"var_dict": var_dict, "var_names": var_names, "var_ord": var_ord}
    return _dataset_info_cache[dataset]


def run_benchmark(model, ann_model, few_shot, cache_dir=None):
    gold = pd.read_parquet(GOLD_LABELED_PATH)
    n_total_rows = len(gold)
    gold = gold[gold["claude_answer"].notna()].reset_index(drop=True)
    if len(gold) < n_total_rows:
        print(f"  [skip] {n_total_rows - len(gold)} gold rows have no claude_answer (excluded)")

    rows = []
    for (dataset, variable), group in gold.groupby(["dataset", "variable"]):
        info = dataset_info(dataset)
        var_dict = {variable: info["var_dict"][variable]}
        var_names = {variable: info["var_names"].get(variable, variable)}
        var_ord = {variable: info["var_ord"][variable]}

        cache_path = None
        if cache_dir is not None:
            tag = "fewshot" if few_shot else "zeroshot"
            cache_path = os.path.join(cache_dir, f"{ann_model}_{tag}_{dataset}_{variable}_ann.parquet")

        df_ann = annotate_data(
            model, None, None,
            [group["story"].tolist()],  # single attempt, no retries
            var_dict, var_names, var_ord,
            engine="vllm",
            cache_path=cache_path,
            dataset=dataset,
            few_shot=few_shot,
        )

        predicted = df_ann[variable].tolist()
        for gold_id, gen_model, truth, pred in zip(
            group["gold_id"], group["gen_model"], group["claude_answer"], predicted
        ):
            pred_norm = "NA" if pd.isna(pred) else str(pred)
            rows.append({
                "gold_id": gold_id, "dataset": dataset, "variable": variable,
                "gen_model": gen_model, "claude_answer": truth, "predicted": pred_norm,
                "correct": pred_norm == truth,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_model", type=str, required=True)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--out_dir", type=str, default="data/gold_test/benchmark")
    args = parser.parse_args()

    tag = f"{args.ann_model}_{'fewshot' if args.few_shot else 'zeroshot'}"
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading {args.ann_model} (vLLM)...")
    model = get_vllm_model(MODEL_PATHS[args.ann_model])

    results = run_benchmark(model, args.ann_model, args.few_shot, cache_dir=cache_dir)

    score = int(results["correct"].sum())
    total = len(results)

    results.to_csv(os.path.join(args.out_dir, f"{tag}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"{tag}_score.txt"), "w") as f:
        f.write(f"{score}/{total}\n")

    print(f"\n{args.ann_model} ({'few-shot' if args.few_shot else 'zero-shot'}): {score}/{total}")

    # quick per-dataset breakdown, useful without opening the CSV
    for dataset, sub in results.groupby("dataset"):
        print(f"  {dataset}: {int(sub['correct'].sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
