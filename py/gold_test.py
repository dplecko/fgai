# Build a gold test set from Qwen2.5-72B vs Command R+ 104B annotator
# disagreements (attempt 1, group ""), then label it with Claude Opus 5.
#
# Usage:
#   python3 py/gold_test.py build                 # sample 300 disagreements, save gold_test_set.parquet
#   python py/gold_test.py submit                 # submit the Claude Batches job
#   python3 py/gold_test.py collect [--wait]        # poll / fetch results, save gold_test_labeled.parquet
#
# `build` only needs pandas/pyarrow (matches R/scripts/annotator-agreement.R's
# data reading approach). `submit`/`collect` additionally need the `anthropic`
# package and network access to api.anthropic.com -- run those from a node
# with outbound internet, not necessarily the GPU/vLLM container.
import argparse
import json
import os
os.environ["ANTHROPIC_API_KEY"] = "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"
import re
import string
import time

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


GENERAL_RULES = (
    "1. If there are multiple people or narratives, focus only on the first one.\n"
    "2. If there is duplicate or contradictory information about the person, answer NA.\n"
    "3. If the answer is not reasonably clear, answer NA rather than guessing."
)

# Dataset/variable-specific rules, appended to GENERAL_RULES when applicable.
# Extend this dict as more quirks like this one turn up during review.
SPECIAL_RULES = {
    ("census_income", "race"): (
        "4. Special rule for race: \"Hispanic\" is an ethnicity, not one of the race "
        "categories listed below. \"Hispanic\" alone, with no other race stated -> NA. "
        "\"Hispanic\" plus a stated race category -> use that race category. Two distinct "
        "race categories stated (e.g. White and Black) -> \"mix\". A single race stated that "
        "isn't one of the other listed categories (e.g. Middle Eastern, Moroccan) -> \"other\"."
    ),
}


def build_gold_prompt(dataset, variable, story, var_name, levels):
    options = "\n".join(f"- {lvl}" for lvl in levels)
    rules = GENERAL_RULES
    special = SPECIAL_RULES.get((dataset, variable))
    if special:
        rules = rules + "\n" + special

    return (
        "Consider the following story:\n\n"
        f"{story}\n\n"
        f"Based on the story, determine the person's {var_name}.\n\n"
        f"Valid options:\n{options}\n\n"
        f"Rules:\n{rules}\n\n"
        "Also identify who the answer refers to (e.g. \"the narrator\", \"the man described "
        "in the first paragraph\")."
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
            "claude_prompt": build_gold_prompt(r["dataset"], r["variable"], story, var_name, levels),
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
        if result.result.type != "succeeded":
            errors.append((result.custom_id, result.result.type))
            continue

        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            errors.append((result.custom_id, f"no text block (stop_reason={message.stop_reason})"))
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Most likely cause: max_tokens cut generation off mid-JSON (thinking
            # is on by default on Opus 5 and shares the max_tokens budget with the
            # response) -- stop_reason confirms it. Skip and report, don't crash
            # the whole collection over one bad record.
            errors.append((result.custom_id, f"invalid JSON (stop_reason={message.stop_reason})"))
            continue

        labels[result.custom_id] = {
            "claude_target_person": parsed.get("target_person"),
            "claude_evidence": parsed.get("evidence"),
            "claude_answer": parsed.get("answer"),
        }

    if errors:
        print(f"  {len(errors)} requests could not be parsed:")
        for custom_id, reason in errors:
            print(f"    {custom_id}: {reason}")

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
    p_submit.add_argument("--max_tokens", type=int, default=4096)

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

# import pandas as pd

# def review_disagreements(n, n_wrong, path="data/gold_test/gold_test_labeled.parquet", seed=None):
#     """
#     Print n gold-test rows where exactly n_wrong of {qwen, commandrp} disagree
#     with Claude's answer, pausing after each. n_wrong must be 1 or 2.
#     """
#     assert n_wrong in (1, 2), "n_wrong must be 1 or 2"

#     df = pd.read_parquet(path)
#     df = df[df["claude_answer"].notna()].copy()

#     df["wrong_count"] = (
#         (df["qwen_answer"] != df["claude_answer"]).astype(int)
#         + (df["commandrp_answer"] != df["claude_answer"]).astype(int)
#     )

#     pool = df[df["wrong_count"] == n_wrong]
#     if pool.empty:
#         print(f"No rows with exactly {n_wrong} model(s) disagreeing with Claude.")
#         return

#     sample = pool.sample(n=min(n, len(pool)), random_state=seed)

#     for i, (_, r) in enumerate(sample.iterrows(), 1):
#         print("=" * 70)
#         print(f"[{i}/{len(sample)}] {r['dataset']} / {r['gen_model']} / row {r['row']} / {r['variable']} ({r['category']})")
#         print("-" * 70)
#         print(r["local_prompt"])
#         print("-" * 70)
#         print(f"qwen:       {r['qwen_answer']}")
#         print(f"commandrp:  {r['commandrp_answer']}")
#         print(f"claude:     {r['claude_answer']}   (target: {r['claude_target_person']})")
#         print(f"  evidence: {r['claude_evidence']}")
#         print("=" * 70)

#         cmd = input("[c]ontinue, [q]uit: ").strip().lower()
#         if cmd.startswith("q"):
#             print("Stopped.")
#             break

# gt.loc[(gt["dataset"] == "census_income") & (gt["variable"] == "race")]
# gt.loc[gt.index[158], "claude_target_person"] = "Michael, 35 year old White male"
# gt.loc[gt.index[158], "claude_evidence"] = "He holds a Bachelor's degree"
# gt.loc[gt.index[158], "claude_answer"] = "Bachelor's degree"
# gt.loc[gt.index[55], "claude_answer"] = 'Bachelor’s or higher'


# gt = pd.read_parquet("data/gold_test/gold_test_labeled.parquet")
# print(gt["local_prompt"][127])


# gt.loc[gt.index[127], "claude_target_person"] = "David, 45 year-old from US"
# gt.loc[gt.index[127], "claude_evidence"] = "He is an Asian White individual"
# gt.loc[gt.index[127], "claude_answer"] = "Multiple"

# # gt.to_parquet("data/gold_test/gold_test_labeled.parquet")

def check_claude_answers(path=LABELED_PATH):
    """
    Verify every claude_answer is exactly one of the valid options (the
    variable's levels + 'NA') for its row. Flags exact mismatches, and
    separately flags case-only mismatches -- structured outputs' enum
    constraint should make these impossible, so a nonzero count here is
    worth reporting, not just silently normalizing away.
    """
    df = pd.read_parquet(path)
    mismatches = []
    for _, r in df.iterrows():
        if pd.isna(r["claude_answer"]):
            continue
        levels = dataset_info(r["dataset"])["var_dict"][r["variable"]]
        valid = list(levels) + ["NA"]
        if r["claude_answer"] in valid:
            continue

        lower_map = {v.lower(): v for v in valid}
        close = lower_map.get(str(r["claude_answer"]).lower())

        mismatches.append({
            "gold_id": r["gold_id"],
            "dataset": r["dataset"],
            "variable": r["variable"],
            "claude_answer": r["claude_answer"],
            "closest_valid_option": close,
            "valid_options": valid,
        })

    mismatch_df = pd.DataFrame(mismatches)
    if mismatch_df.empty:
        print("All claude_answer values match a valid option exactly.")
    else:
        n_case_only = mismatch_df["closest_valid_option"].notna().sum()
        print(f"{len(mismatch_df)} claude_answer values don't exactly match a valid option "
              f"({n_case_only} case-only, {len(mismatch_df) - n_case_only} with no close match at all).")
    return mismatch_df


# cenra = gt.loc[(gt["dataset"] == "census_income") & (gt["variable"] == "race")]
# for i, (_, r) in enumerate(cenra.iterrows(), 1):
#         print("=" * 70)
#         print(f"[{i}/{len(cenra)}] {r['dataset']} / {r['gen_model']} / row {r['row']} / {r['variable']} ({r['category']})")
#         print("-" * 70)
#         print(r["local_prompt"])
#         print("-" * 70)
#         print(f"qwen:       {r['qwen_answer']}")
#         print(f"commandrp:  {r['commandrp_answer']}")
#         print(f"claude:     {r['claude_answer']}   (target: {r['claude_target_person']})")
#         print(f"  evidence: {r['claude_evidence']}")
#         print("=" * 70)
#         cmd = input("[c]ontinue, [q]uit: ").strip().lower()
#         if cmd.startswith("q"):
#             print("Stopped.")
#             break
