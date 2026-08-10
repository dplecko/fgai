# Build a gold test set from Qwen2.5-72B vs Command R+ 104B annotator
# disagreements (attempt 1, group ""), then label it with Claude Opus 5.
#
# Usage:
#   python3 py/gold-test.py build                 # sample 300 disagreements, save gold_test_set.parquet
#   python3 py/gold-test.py submit                 # submit the Claude Batches job
#   python3 py/gold-test.py collect [--wait]        # poll / fetch results, save gold_test_labeled.parquet
#
# `build` only needs pandas/pyarrow (matches R/scripts/annotator-agreement.R's
# data reading approach). `submit`/`collect` additionally need the `anthropic`
# package and network access to api.anthropic.com -- run those from a node
# with outbound internet, not necessarily the GPU/vLLM container.
import argparse
import json
import os
import re
import string
import time

import pandas as pd

from py.data_helpers import load_data

DATASETS = ["nsduh", "brfss", "census_income"]
GEN_MODELS = [
    "llama3_8b", "ministral3_8b", "gemma3_4b", "qwen35_9b", "deepseek_7b", "phi4",
    "qwen35_27b", "gemma3_27b", "deepseek_r1", "llama3_70b",
]
ANN_A = "qwen25_72b"
ANN_B = "commandrp_104b"
GROUP = ""  # matches scripts/annotator-agreement.R: every variable is inferred
N_NA = 100
N_VALUE = 200
SEED = 2025
OUT_DIR = "data/gold_test"
GOLD_SET_PATH = os.path.join(OUT_DIR, "gold_test_set.parquet")
GOLD_SET_CSV = os.path.join(OUT_DIR, "gold_test_set.csv")
BATCH_ID_PATH = os.path.join(OUT_DIR, "batch_id.txt")
LABELED_PATH = os.path.join(OUT_DIR, "gold_test_labeled.parquet")
LABELED_CSV = os.path.join(OUT_DIR, "gold_test_labeled.csv")

NA_LETTER_RE = re.compile(r"([A-Z])\.\s*Answer not available")

_dataset_info_cache = {}
_gen_text_cache = {}


def dataset_info(dataset):
    if dataset not in _dataset_info_cache:
        _df, var_dict, var_names, _var_ord, _sfm, _context = load_data(dataset)
        _dataset_info_cache[dataset] = {"var_dict": var_dict, "var_names": var_names}
    return _dataset_info_cache[dataset]


def gen_texts(dataset, gen_model):
    key = (dataset, gen_model)
    if key not in _gen_text_cache:
        path = f"data/cache/{dataset}_{gen_model}_{GROUP}_gen.parquet"
        _gen_text_cache[key] = pd.read_parquet(path)["response"].tolist()
    return _gen_text_cache[key]


def decode_letter(letter, levels):
    """Map a response letter to its category text, or 'NA' for the trailing
    'Answer not available' option. None if the letter itself is missing."""
    if not isinstance(letter, str) or len(letter) != 1 or letter not in string.ascii_uppercase:
        return None
    idx = string.ascii_uppercase.index(letter)
    if idx < len(levels):
        return levels[idx]
    return "NA"


def read_ann1(dataset, gen_model, ann_model):
    """Attempt-1 annotation log for one (dataset, gen_model, ann_model), with
    an is_na flag derived the same way as scripts/annotator-agreement.R:
    regex the NA letter straight out of the stored prompt text."""
    path = f"data/cache/{dataset}_{gen_model}_{ann_model}_{GROUP}_ann.parquet"
    if not os.path.exists(path):
        return None
    log = pd.read_parquet(path)
    if "row" not in log.columns:
        log["row"] = log.groupby("variable").cumcount()
    if "attempt" not in log.columns:
        log["attempt"] = 1
    log = log[log["attempt"] == 1].drop_duplicates(subset=["row", "variable"], keep="first")
    na_letter = log["prompt"].str.extract(NA_LETTER_RE)[0]
    log = log.assign(na_letter=na_letter, is_na=log["response"] == na_letter)
    return log[["row", "variable", "response", "is_na", "prompt"]]


def build_disagreements():
    """All attempt-1 (row, variable) disagreements between ANN_A and ANN_B,
    across every (dataset, gen_model) pair, group GROUP only."""
    pools = []
    for dataset in DATASETS:
        for gen_model in GEN_MODELS:
            la = read_ann1(dataset, gen_model, ANN_A)
            lb = read_ann1(dataset, gen_model, ANN_B)
            if la is None or lb is None:
                print(f"    [skip] missing ann_cache for {dataset}/{gen_model}")
                continue
            merged = la.merge(lb, on=["row", "variable"], suffixes=("_a", "_b"))
            merged = merged[merged["response_a"] != merged["response_b"]].copy()
            merged["dataset"] = dataset
            merged["gen_model"] = gen_model
            pools.append(merged)
    if not pools:
        raise RuntimeError(f"No ann_cache files found for {ANN_A} / {ANN_B} -- run ann_only.sh first")
    return pd.concat(pools, ignore_index=True)


def build_gold_prompt(story, var_name, levels):
    options = "\n".join(f"- {lvl}" for lvl in levels)
    return (
        "Consider the following story:\n\n"
        f"{story}\n\n"
        "If there are multiple narratives, focus only on the first one, and identify "
        "who it is about (e.g. \"the narrator\", \"the man described in the first paragraph\").\n\n"
        f"Based on the story, determine the person's {var_name}.\n\n"
        f"Valid options:\n{options}\n\n"
        "If the story does not state this fact, directly or by clear implication, "
        "answer \"NA\" rather than guessing."
    )


def build_schema(levels):
    # target_person -> evidence -> answer: the model identifies the subject and
    # cites its evidence before committing to a category, rather than answering
    # first and justifying after.
    return {
        "type": "object",
        "properties": {
            "target_person": {
                "type": "string",
                "description": "Who in the story this answer refers to.",
            },
            "evidence": {
                "type": "string",
                "description": "The specific sentence or phrase the answer is based on, "
                                "or why nothing in the story supports an answer.",
            },
            "answer": {
                "type": "string",
                "enum": list(levels) + ["NA"],
            },
        },
        "required": ["target_person", "evidence", "answer"],
        "additionalProperties": False,
    }


def cmd_build(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Building attempt-1 disagreement pool for {ANN_A} vs {ANN_B}...")
    pool = build_disagreements()
    n_pairs = pool[["dataset", "gen_model"]].drop_duplicates().shape[0]
    print(f"  {len(pool)} total disagreements across {n_pairs} dataset/gen_model pairs")

    na_pool = pool[pool["is_na_a"] | pool["is_na_b"]]
    value_pool = pool[~pool["is_na_a"] & ~pool["is_na_b"]]
    print(f"  NA-involved disagreements: {len(na_pool)}")
    print(f"  value-only disagreements:  {len(value_pool)}")

    if len(na_pool) < N_NA:
        print(f"  WARNING: only {len(na_pool)} NA disagreements available (< {N_NA})")
    if len(value_pool) < N_VALUE:
        print(f"  WARNING: only {len(value_pool)} value disagreements available (< {N_VALUE})")

    na_sample = na_pool.sample(n=min(N_NA, len(na_pool)), random_state=SEED).assign(category="na_disagreement")
    value_sample = value_pool.sample(n=min(N_VALUE, len(value_pool)), random_state=SEED).assign(category="value_disagreement")
    gold = pd.concat([na_sample, value_sample], ignore_index=True)

    records = []
    for i, r in gold.iterrows():
        info = dataset_info(r["dataset"])
        levels = info["var_dict"][r["variable"]]
        var_name = info["var_names"].get(r["variable"], r["variable"])
        story = gen_texts(r["dataset"], r["gen_model"])[r["row"]]

        records.append({
            "gold_id": f"gold-{i:04d}",
            "dataset": r["dataset"],
            "gen_model": r["gen_model"],
            "row": int(r["row"]),
            "variable": r["variable"],
            "category": r["category"],
            "story": story,
            "local_prompt": r["prompt_a"],
            "claude_prompt": build_gold_prompt(story, var_name, levels),
            "schema": json.dumps(build_schema(levels)),
            "qwen_letter": r["response_a"],
            "qwen_answer": decode_letter(r["response_a"], levels),
            "commandrp_letter": r["response_b"],
            "commandrp_answer": decode_letter(r["response_b"], levels),
        })

    gold_df = pd.DataFrame.from_records(records)
    gold_df.to_parquet(GOLD_SET_PATH, index=False)
    gold_df.drop(columns=["schema"]).to_csv(GOLD_SET_CSV, index=False)
    print(f"Saved {len(gold_df)} gold-test rows to {GOLD_SET_PATH} "
          f"({len(na_sample)} NA, {len(value_sample)} value)")


def cmd_submit(args):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    gold_df = pd.read_parquet(GOLD_SET_PATH)
    client = anthropic.Anthropic()

    requests = []
    for _, r in gold_df.iterrows():
        schema = json.loads(r["schema"])
        requests.append(Request(
            custom_id=r["gold_id"],
            params=MessageCreateParamsNonStreaming(
                model=args.model,
                max_tokens=args.max_tokens,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": r["claude_prompt"]}],
            ),
        ))

    batch = client.messages.batches.create(requests=requests)
    with open(BATCH_ID_PATH, "w") as f:
        f.write(batch.id)
    print(f"Submitted batch {batch.id} ({len(requests)} requests, model={args.model})")
    print("Check status with: python3 py/gold-test.py collect")


def cmd_collect(args):
    import anthropic

    with open(BATCH_ID_PATH) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch {batch_id}: {batch.processing_status} "
              f"(succeeded={batch.request_counts.succeeded}, "
              f"errored={batch.request_counts.errored}, "
              f"processing={batch.request_counts.processing})")
        if batch.processing_status == "ended" or not args.wait:
            break
        time.sleep(args.poll_interval)

    if batch.processing_status != "ended":
        print("Batch not finished yet -- re-run collect later (or with --wait).")
        return

    labels = {}
    errors = []
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            text = next(b.text for b in result.result.message.content if b.type == "text")
            parsed = json.loads(text)
            labels[result.custom_id] = {
                "claude_target_person": parsed.get("target_person"),
                "claude_evidence": parsed.get("evidence"),
                "claude_answer": parsed.get("answer"),
            }
        else:
            errors.append((result.custom_id, result.result.type))

    if errors:
        print(f"  {len(errors)} requests did not succeed: {errors[:10]}")

    gold_df = pd.read_parquet(GOLD_SET_PATH)
    label_df = pd.DataFrame.from_dict(labels, orient="index").reset_index().rename(columns={"index": "gold_id"})
    labeled = gold_df.merge(label_df, on="gold_id", how="left")

    labeled.to_parquet(LABELED_PATH, index=False)
    labeled.drop(columns=["schema"]).to_csv(LABELED_CSV, index=False)
    print(f"Saved {len(labeled)} labeled rows to {LABELED_PATH}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--model", type=str, default="claude-opus-5")
    p_submit.add_argument("--max_tokens", type=int, default=1024)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--wait", action="store_true", help="poll until the batch ends")
    p_collect.add_argument("--poll_interval", type=int, default=60)

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
