# One-off test: does asking the model to identify the target person AND
# answer in a single structured-output call (rather than annotate_data's
# single constrained-letter decode) help a model that struggles to "focus on
# the first person" (llama3_70b, per manual review of goldtest_benchmark.py
# results)?
#
# One generate() call per row: the prompt asks for one line of JSON
# ({"target_person": ..., "answer": "<letter>"}), mirroring the Claude gold
# judge's target_person -> evidence -> answer structure (reasons about who
# before committing to an answer), but in a single turn. Not enforced via
# vLLM guided/schema decoding (unverified API surface for this vLLM version)
# -- parsed from free text with a forgiving fallback instead.
#
# Same 300-row gold set, same scoring against claude_answer, same rules
# content (reuses SPECIAL_RULES from generation.py) as the regular benchmark,
# so results are directly comparable.
#
# Usage (inside the fgai container, needs a GPU):
#   python3 -m py.goldtest_structured --ann_model llama3_70b
import argparse
import os

import pandas as pd

from py.model_load import MODEL_PATHS, get_vllm_model
from py.generation import prepare_answers, SPECIAL_RULES, ANNOTATION_RULES
from py.few_shot_examples import FEW_SHOT_EXAMPLES
from py.few_shot_helpers import render_demos_section, parse_json_answer
from py.data_helpers import load_data

# SPECIAL_RULES (including the education rule) now lives in generation.py --
# imported directly above, no local override, so this and the production
# pipeline can't drift apart on rules content.

GOLD_LABELED_PATH = "data/gold_test/gold_test_labeled.parquet"
OUT_DIR = "data/gold_test/benchmark"

# Rows judged, after manual review, not to be a fair test of the annotator --
# either a Claude labeling error (since fixed in the parquet) or a story too
# ambiguous for a careful judge to call. Positional indices into the gold set
# after the claude_answer.notna() filter (same ordering load_gold below and
# every benchmark script uses). Confirmed 2026-08-13; drops 300 -> 290 rows.
EXCLUDED_ROWS = [3, 113, 151, 181, 186, 213, 240, 178, 252, 274]


def load_gold():
    """Load the gold set with the standard notna + excluded-row filtering
    applied, so every benchmark script scores against the same rows."""
    gold = pd.read_parquet(GOLD_LABELED_PATH)
    n_total_rows = len(gold)
    gold = gold[gold["claude_answer"].notna()].reset_index(drop=True)
    if len(gold) < n_total_rows:
        print(f"  [skip] {n_total_rows - len(gold)} gold rows have no claude_answer (excluded)")
    gold = gold.drop(index=EXCLUDED_ROWS).reset_index(drop=True)
    print(f"  [skip] {len(EXCLUDED_ROWS)} gold rows excluded as not a fair test (ambiguous/fixed Claude errors)")
    return gold


_dataset_info_cache = {}


def dataset_info(dataset):
    if dataset not in _dataset_info_cache:
        _df, var_dict, var_names, _var_ord, _sfm, _context = load_data(dataset)
        _dataset_info_cache[dataset] = {"var_dict": var_dict, "var_names": var_names}
    return _dataset_info_cache[dataset]


def build_prompt(story, var_name, levels, extra_rule=None, few_shot_examples=None):
    """
    few_shot_examples: optional dict of the demo entries for this
    (dataset, variable) pair (FEW_SHOT_EXAMPLES[key]["examples"]), each with
    "story"/"explanation"/"answer". When given, demos are rendered with a
    3-field JSON answer (adds "explanation") and the real query adopts the
    same 3-field schema, story placed *last* so the whole preamble (rules/
    demos/question/answer options/instructions) is a byte-identical,
    prefix-cacheable prefix across every row -- matches generation.py's
    prep_ann_prompt exactly (rules and demo-rendering are shared via
    ANNOTATION_RULES / py.few_shot_helpers, not duplicated).

    When None (default), behavior is unchanged from before few-shot support
    was added: 2-field JSON, no demos, story first (not cacheable) -- kept
    as-is so other in-flight benchmark comparisons stay unaffected.
    """
    rules = ANNOTATION_RULES
    if extra_rule:
        rules += extra_rule + "\n"
    answer_key, answer_mapping = prepare_answers(levels)

    if few_shot_examples:
        demos_section = render_demos_section(few_shot_examples, var_name, answer_key, answer_mapping)
        json_instruction = (
            "First identify who the story is about, briefly explain your reasoning, then "
            "choose the correct option letter. Respond with a single line of JSON in "
            "exactly this form, no other text:\n"
            '{"target_person": "<who the story is about>", "explanation": "<brief reasoning>", "answer": "<letter>"}'
        )
        prompt = (
            f"{rules}\n"
            f"{demos_section}"
            f"Question: determine the person's {var_name}.\n\n"
            f"{answer_key}\n\n"
            f"{json_instruction}\n\n"
            f"<story>\n{story}\n</story>"
        )
    else:
        # Unchanged from before few-shot support existed -- no <story> tags
        # here either, so every other caller of build_prompt (in-process
        # goldtest_structured.py runs on other models) is unaffected. The
        # tagged/few-shot format is opt-in via few_shot_examples, currently
        # wired up only for the llama3_405b HTTP benchmark.
        prompt = (
            "Consider the following story:\n\n"
            f"{story}\n\n"
            f"{rules}\n"
            f"Question: determine the person's {var_name}.\n\n"
            f"{answer_key}\n\n"
            "First identify who the story is about, then choose the correct option letter. "
            "Respond with a single line of JSON in exactly this form, no other text:\n"
            '{"target_person": "<who the story is about>", "answer": "<letter>"}'
        )
    return prompt, answer_mapping


def main():
    from vllm import SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_model", type=str, required=True)
    parser.add_argument("--few_shot", action="store_true",
                        help="Prepend the hand-written FEW_SHOT_EXAMPLES demos per (dataset, variable) "
                             "and switch to the 3-field JSON schema (adds \"explanation\"). Falls back to "
                             "zero-shot for any (dataset, variable) pair not covered by FEW_SHOT_EXAMPLES.")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Defaults to 150 (zero-shot) or 300 (--few_shot, to leave room for the explanation field)")
    args = parser.parse_args()
    max_tokens = args.max_tokens or (300 if args.few_shot else 150)

    gold = load_gold()

    print(f"Loading {args.ann_model} (vLLM)...")
    model = get_vllm_model(MODEL_PATHS[args.ann_model])
    tokenizer = model.get_tokenizer()
    model_id = model.llm_engine.model_config.model.lower()
    template_kwargs = {"enable_thinking": False} if ("qwen" in model_id or "glm" in model_id) else {}

    predictions = pd.Series(index=gold.index, dtype=object)
    target_persons = pd.Series(index=gold.index, dtype=object)
    explanations = pd.Series(index=gold.index, dtype=object)
    parse_failed = pd.Series(False, index=gold.index)
    raw_responses = pd.Series(index=gold.index, dtype=object)

    for (dataset, variable), group in gold.groupby(["dataset", "variable"]):
        info = dataset_info(dataset)
        levels = info["var_dict"][variable]
        var_name = info["var_names"].get(variable, variable)
        extra_rule = SPECIAL_RULES.get((dataset, variable))

        few_shot_examples = None
        if args.few_shot:
            entry = FEW_SHOT_EXAMPLES.get((dataset, variable))
            if entry is None:
                print(f"  [few_shot] no demos for ({dataset}, {variable}) -- falling back to zero-shot for this group")
            else:
                try:
                    few_shot_examples = entry["examples"]
                    build_prompt(group.iloc[0]["story"], var_name, levels, extra_rule, few_shot_examples)
                except ValueError as e:
                    print(f"  [few_shot] demos for ({dataset}, {variable}) don't match live levels ({e}) -- falling back to zero-shot")
                    few_shot_examples = None

        prompts, answer_mapping = [], None
        for row in group.itertuples():
            p, am = build_prompt(row.story, var_name, levels, extra_rule, few_shot_examples)
            prompts.append(p)
            answer_mapping = am  # identical across the group (fixed dataset/variable)

        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True, return_dict=False, add_generation_prompt=True,
                **template_kwargs,
            )
            for p in prompts
        ]
        sp = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = model.generate([{"prompt_token_ids": p} for p in chat_prompts], sp)

        for idx, out in zip(group.index, outputs):
            text = out.outputs[0].text.strip()
            raw_responses[idx] = text
            letter, target_person, explanation = parse_json_answer(text)
            target_persons[idx] = target_person
            explanations[idx] = explanation
            if letter is None:
                parse_failed[idx] = True
                predictions[idx] = None
            else:
                predictions[idx] = answer_mapping.get(letter)  # None (-> "NA") if not a valid letter

    gold["target_person"] = target_persons
    gold["explanation"] = explanations
    gold["predicted"] = predictions.apply(lambda v: "NA" if v is None else str(v))
    gold["parse_failed"] = parse_failed
    gold["correct"] = gold["predicted"] == gold["claude_answer"]
    gold["raw_response"] = raw_responses

    score = int(gold["correct"].sum())
    total = len(gold)
    n_parse_failed = int(gold["parse_failed"].sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.ann_model}_structured" + ("_fewshot" if args.few_shot else "")
    gold[["gold_id", "dataset", "variable", "gen_model", "story", "target_person",
          "explanation", "claude_answer", "predicted", "parse_failed", "correct", "raw_response"]].to_csv(
        os.path.join(OUT_DIR, f"{tag}.csv"), index=False
    )
    with open(os.path.join(OUT_DIR, f"{tag}_score.txt"), "w") as f:
        f.write(f"{score}/{total}\n")

    print(f"\n{args.ann_model} (structured, single-call{', few-shot' if args.few_shot else ''}): {score}/{total}"
          f" ({n_parse_failed} parse failures, counted as incorrect)")
    for dataset, sub in gold.groupby("dataset"):
        print(f"  {dataset}: {int(sub['correct'].sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
