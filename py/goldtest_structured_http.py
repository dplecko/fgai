# HTTP-client variant of goldtest_structured.py, for use against a
# multi-node `vllm serve` deployment (distributed_executor_backend=mp)
# instead of an in-process LLM() + Ray. Reuses goldtest_structured.py's pure
# prompt-building/parsing/scoring logic unchanged (none of it touches the
# vLLM engine directly) -- only the generation call itself is replaced with
# concurrent HTTP requests against the server's OpenAI-compatible API.
#
# Usage (run from anywhere with network access to the server, e.g. the head
# node inside the container once `vllm serve` is up):
#   python3 -m py.goldtest_structured_http \
#       --server_url http://<HEAD_IP>:8000/v1 \
#       --ann_model llama3_405b --served_model_name llama3_405b
import argparse
import concurrent.futures
import json
import os

import pandas as pd
from openai import OpenAI

from py.few_shot_examples import FEW_SHOT_EXAMPLES
from py.goldtest_structured import (
    OUT_DIR, JSON_RE, SPECIAL_RULES,
    build_prompt, parse_letter, dataset_info, load_gold,
)


def query_one(client, served_model_name, prompt, template_kwargs, max_tokens):
    extra_body = {"chat_template_kwargs": template_kwargs} if template_kwargs else {}
    resp = client.chat.completions.create(
        model=served_model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""


def main():
    from py.model_load import MODEL_PATHS

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, required=True,
                        help="Base URL of the running vllm serve instance, e.g. http://<HEAD_IP>:8000/v1")
    parser.add_argument("--ann_model", type=str, required=True,
                        help="Used to tag output files and (via name matching) detect qwen/glm for enable_thinking")
    parser.add_argument("--served_model_name", type=str, default=None,
                        help="Model name as registered with the server; defaults to MODEL_PATHS[--ann_model]")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--few_shot", action="store_true",
                        help="Prepend the 4 hand-written FEW_SHOT_EXAMPLES demos per (dataset, variable) "
                             "and switch to the 3-field JSON schema (adds \"explanation\"). Falls back to "
                             "zero-shot for any (dataset, variable) pair not covered by FEW_SHOT_EXAMPLES.")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Defaults to 150 (zero-shot) or 300 (--few_shot, to leave room for the explanation field)")
    args = parser.parse_args()

    served_model_name = args.served_model_name or MODEL_PATHS[args.ann_model]
    client = OpenAI(base_url=args.server_url, api_key="not-needed")
    template_kwargs = {"enable_thinking": False} if ("qwen" in args.ann_model.lower() or "glm" in args.ann_model.lower()) else {}
    max_tokens = args.max_tokens or (300 if args.few_shot else 150)

    gold = load_gold()

    # Build every prompt up front, same per-(dataset, variable) rule/level
    # lookup as goldtest_structured.py -- just not grouped into batched
    # generate() calls, since each row is now its own HTTP request.
    prompts = [None] * len(gold)
    answer_mappings = [None] * len(gold)
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
                    # dry-run once against the live levels/mapping so a stale
                    # demo answer (few_shot_examples.py levels drifted from
                    # the live var_dict) fails loud here, not mid-request
                    build_prompt(group.iloc[0]["story"], var_name, levels, extra_rule, few_shot_examples)
                except ValueError as e:
                    print(f"  [few_shot] demos for ({dataset}, {variable}) don't match live levels ({e}) -- falling back to zero-shot")
                    few_shot_examples = None

        for row in group.itertuples():
            p, am = build_prompt(row.story, var_name, levels, extra_rule, few_shot_examples)
            prompts[row.Index] = p
            answer_mappings[row.Index] = am

    print(f"Querying {len(gold)} rows against {args.server_url} (concurrency={args.concurrency})...")
    texts = [""] * len(gold)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(query_one, client, served_model_name, prompts[i], template_kwargs, max_tokens): i
            for i in range(len(gold))
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                texts[i] = future.result()
            except Exception as e:
                print(f"  [error] row {i}: {e}")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(gold)} done")

    target_persons = pd.Series(index=gold.index, dtype=object)
    explanations = pd.Series(index=gold.index, dtype=object)
    predictions = pd.Series(index=gold.index, dtype=object)
    parse_failed = pd.Series(False, index=gold.index)

    for i in range(len(gold)):
        text = (texts[i] or "").strip()
        m = JSON_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(0))
                target_persons[i] = parsed.get("target_person")
                explanations[i] = parsed.get("explanation")
            except (json.JSONDecodeError, AttributeError):
                pass
        letter = parse_letter(text)
        if letter is None:
            parse_failed[i] = True
            predictions[i] = None
        else:
            predictions[i] = answer_mappings[i].get(letter)

    gold["target_person"] = target_persons
    gold["explanation"] = explanations
    gold["predicted"] = predictions.apply(lambda v: "NA" if v is None else str(v))
    gold["parse_failed"] = parse_failed
    gold["correct"] = gold["predicted"] == gold["claude_answer"]
    # kept mainly so parse_failed rows are still auditable after the fact --
    # without it there was no way to see what the model actually said
    gold["raw_response"] = texts

    score = int(gold["correct"].sum())
    total = len(gold)
    n_parse_failed = int(gold["parse_failed"].sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.ann_model}_structured_http" + ("_fewshot" if args.few_shot else "")
    gold[["gold_id", "dataset", "variable", "gen_model", "story", "target_person",
          "explanation", "claude_answer", "predicted", "parse_failed", "correct", "raw_response"]].to_csv(
        os.path.join(OUT_DIR, f"{tag}.csv"), index=False
    )
    with open(os.path.join(OUT_DIR, f"{tag}_score.txt"), "w") as f:
        f.write(f"{score}/{total}\n")

    print(f"\n{args.ann_model} (structured, HTTP/mp{', few-shot' if args.few_shot else ''}): {score}/{total}"
          f" ({n_parse_failed} parse failures, counted as incorrect)")
    for dataset, sub in gold.groupby("dataset"):
        print(f"  {dataset}: {int(sub['correct'].sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
