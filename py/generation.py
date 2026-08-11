import os
import re
import string
import torch
from tqdm import tqdm
import pandas as pd

from py.few_shot_examples import FEW_SHOT_EXAMPLES


def known_facts(kvar_dict):

    start = "known facts to be mentioned:\n"
    # for each variable, mention its name and give the categories
    var_lines = []
    for var, val in kvar_dict.items():
        var_line = f"- {var} = {val[0]}"
        var_lines.append(var_line)

    return start + "\n".join(var_lines)


def unknown_facts(uvar_dict):

    start = "unknown facts to be mentioned:\n"
    # for each variable, mention its name and give the categories
    var_lines = []
    for var, categories in uvar_dict.items():
        cat_str = ", ".join(categories)
        var_line = f"- {var} (possible values: {cat_str})"
        var_lines.append(var_line)

    return start + "\n".join(var_lines)


# make the variable list in prompt
def varlist_to_prompt(var_dict, var_names, context=None, style="story"):

    ctx_line = f"CONTEXT: {context}\n" if context else ""
    if style == "record":
        rule1 = "Write the facts as a bulleted list, one bullet per fact.\n"
        rule2 = "2) Do NOT include headings, narrative prose, analysis, or any text outside the list.\n"
        prompt_end = "OUTPUT FORMAT:\n" "- (fact 1)\n" "- (fact 2)\n" "...\n"
    else:
        rule1 = "Write a single narrative enclosed in <story>...</story>.\n"
        rule2 = "2) Do NOT include headings, lists, analysis, or any text outside the tags.\n"
        prompt_end = "OUTPUT FORMAT:\n" "<story>\n" "(your narrative here)\n" "</story>\n"
    prompt_start = (
        "You are a data generator. Follow the rules strictly.\n"
        + ctx_line +
        "RULES:\n"
        f"1) {rule1}"
        f"{rule2}"
        "3) Mention ALL facts given below exactly once ({}).\n"
        "4) Keep it under 200 words.\n\n"
    )

    prompt_start = prompt_start.format(
        ", ".join(var_names[var] for var in var_dict.keys())
    )

    # split var_dict into known variables (single level) and unknown variables (multi-level)
    known_vars = {var: cats for var, cats in var_dict.items() if len(cats) == 1}
    unknown_vars = {var: cats for var, cats in var_dict.items() if len(cats) > 1}

    known = known_facts(known_vars) if len(known_vars) > 0 else ""
    unknown = unknown_facts(unknown_vars) if len(unknown_vars) > 0 else ""

    prompt = prompt_start + "\n" + known + "\n" + unknown + "\n\n" + prompt_end
    return prompt


def extract_tag(text: str, tag: str = "story") -> str:
    # Reasoning-capable models may emit a thinking block before the requested
    # response. It is not part of the generated record.
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: if the model disobeys, remove only the requested wrapper and
    # known headings. Do not strip arbitrary <...> text: records can contain
    # comparisons such as "income < $10,000" and "age > 30".
    text = re.sub(rf"(?i)</?{tag}\s*>", "", text)
    text = re.sub(r"(?is)^#+.*?$", "", text)  # markdown headers
    text = re.sub(r"(?is)^(analyzing errors|analysis).*?$", "", text).strip()
    return text.strip()


def _model_id(model=None, tokenizer=None) -> str:
    """Best-effort model identifier for model-specific chat-template flags."""
    if model is not None:
        try:
            return str(model.llm_engine.model_config.model).lower()
        except AttributeError:
            pass
        try:
            return str(model.config.name_or_path).lower()
        except AttributeError:
            pass
    return str(getattr(tokenizer, "name_or_path", "")).lower()


def _chat_template_kwargs(model_id: str) -> dict:
    """Disable emitted reasoning because annotation deliberately returns one token."""
    if "qwen" in model_id or "glm" in model_id:
        return {"enable_thinking": False}
    if "mistral" in model_id:
        return {"reasoning_effort": "none"}
    return {}


def _apply_transformers_chat_template(tokenizer, prompts, template_kwargs=None):
    """Render user prompts with the tokenizer's native chat template."""
    template_kwargs = template_kwargs or {}
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for prompt in prompts
    ]
    # The rendered strings already contain all template special tokens.
    return tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )


def _single_token_answer_ids(tokenizer, letters):
    """Return a validated one-token ID for every annotation answer letter."""
    letter_to_id = {}
    for letter in letters:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Answer label {letter!r} must encode to exactly one token; got {token_ids}. "
                "Choose different labels for this tokenizer."
            )
        letter_to_id[letter] = token_ids[0]

    if len(set(letter_to_id.values())) != len(letter_to_id):
        raise ValueError(f"Answer labels do not have unique token IDs: {letter_to_id}")
    return letter_to_id


def _save_parquet(records, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


@torch.inference_mode()
def gen_data_batched(
    df=None,
    var_dict=None,
    var_names=None,
    context=None,
    nsamp: int = 100,
    model=None,
    tokenizer=None,
    device=None,
    max_new_tokens: int = 256,
    temperature: float = 1,
    top_p: float = 1.0,
    batch_size: int = 16,  # used by transformers engine only
    engine: str = "transformers",
    cache_path: str = None,
    style: str = "story",
):
    # 0) build prompts
    if df is not None:
        nsamp = df.shape[0]
        known_vars = df.columns.tolist()
        prompts = []
        for i in range(nsamp):
            dyn_dict = var_dict.copy()
            for var in known_vars:
                dyn_dict[var] = [str(df.loc[i, var])]
            prompts.append(varlist_to_prompt(dyn_dict, var_names, context=context, style=style))
    else:
        prompt = varlist_to_prompt(var_dict, var_names, context=context, style=style)
        prompts = [prompt for _ in range(nsamp)]

    out_texts = []

    if engine == "vllm":
        from vllm import SamplingParams
        model_id = _model_id(model=model)
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        vllm_tok = model.get_tokenizer()
        template_kwargs = _chat_template_kwargs(model_id)

        chat_prompts = [
            vllm_tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                return_dict=False,  # add
                add_generation_prompt=True,
                **template_kwargs,
            )
            for p in prompts
        ]
        vllm_inputs = [{"prompt_token_ids": p} for p in chat_prompts]
        outputs = model.generate(vllm_inputs, sp)

        for output in outputs:
            raw = output.outputs[0].text.strip()
            story = extract_tag(raw)
            words = story.split()
            if len(words) > 200:
                story = " ".join(words[:200]).rstrip()
            out_texts.append(story)

    else:  # transformers
        model_id = _model_id(model=model, tokenizer=tokenizer)
        template_kwargs = _chat_template_kwargs(model_id)
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating (batched)"):
            batch_prompts = prompts[i : i + batch_size]

            enc = _apply_transformers_chat_template(
                tokenizer,
                batch_prompts,
                template_kwargs,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_width = input_ids.shape[1]

            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=temperature > 0,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                pad_token_id=getattr(
                    tokenizer,
                    "pad_token_id",
                    getattr(tokenizer, "eos_token_id", None),
                ),
                use_cache=True,
            )
            if temperature > 0:
                generation_kwargs.update(temperature=temperature, top_p=top_p)
            gen = model.generate(**generation_kwargs)

            for j in range(gen.size(0)):
                seq = gen[j, prompt_width:]
                raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
                story = extract_tag(raw)
                words = story.split()
                if len(words) > 200:
                    story = " ".join(words[:200]).rstrip()
                out_texts.append(story)

            del enc, input_ids, attention_mask, gen
            torch.cuda.empty_cache()

    if cache_path is not None:
        records = [{"prompt": p, "response": t} for p, t in zip(prompts, out_texts)]
        _save_parquet(records, cache_path)

    return out_texts


# helper functions
NA_LABEL = "Answer not available"


def prepare_answers(levels, allow_na=True):
    """
    Prepare the answers for the model.
    :param levels: A list of possible answers.
    :param allow_na: append an "Answer not available" option mapped to None.
    :return: (answer_key text, mapping of letter -> value). The NA letter (if
        present) maps to None, even though its displayed text is NA_LABEL.
    """
    display = list(levels) + ([NA_LABEL] if allow_na else [])
    if len(display) > 26:
        raise ValueError("Supports up to 26 items (A-Z)")

    letters = string.ascii_uppercase  # 'A', 'B', ...
    answer_key = "\n".join(f"{letters[i]}. {v}" for i, v in enumerate(display))

    mapping = {letters[i]: item for i, item in enumerate(display)}
    if allow_na:
        mapping[letters[len(levels)]] = None

    return answer_key, mapping


ANNOTATION_RULES = (
    "Rules:\n"
    "1. If there are multiple people or narratives, focus only on the first one.\n"
    "2. If there is duplicate or contradictory information about the person, answer NA.\n"
    "3. If the answer is not reasonably clear, answer NA rather than guessing.\n"
)

# Dataset/variable-specific rules, appended to ANNOTATION_RULES when
# applicable (looked up by annotate_data via (dataset, var), same key shape
# as FEW_SHOT_EXAMPLES). Mirrors SPECIAL_RULES in gold-test.py's Claude
# prompt, so the pipeline annotators and the Claude judge follow the same
# rubric. Extend as more dataset/variable quirks like this one turn up.
SPECIAL_RULES = {
    ("census_income", "race"): (
        "4. Special rule for race: \"Hispanic\" is an ethnicity, not one of the race "
        "categories listed below. \"Hispanic\" alone, with no other race stated -> NA. "
        "\"Hispanic\" plus a stated race category -> use that race category. Two distinct "
        "race categories stated (e.g. White and Black) -> \"mix\". A single race stated that "
        "isn't one of the other listed categories (e.g. Middle Eastern, Moroccan) -> \"other\"."
    ),
}


def _render_few_shot_example(story, var_name, levels, answer, extra_rule=None):
    """One demo Q&A block: same prep_ann_prompt format, completed with the
    example's known-correct letter."""
    prompt_text, answer_mapping = prep_ann_prompt(story, var_name, levels, extra_rule=extra_rule)
    if answer == "NA":
        letter = string.ascii_uppercase[len(levels)]
    else:
        letter = next(l for l, v in answer_mapping.items() if v == answer)
    return f"{prompt_text}{letter}\n\n"


def prep_ann_prompt(text, var_name, levels, few_shot_examples=None, extra_rule=None):
    """
    Prepare the prompt for the model.
    :param text: The text to annotate.
    :param levels: A list of possible answers.
    :param few_shot_examples: optional entry from
        py/few_shot_examples.FEW_SHOT_EXAMPLES (with "var_name"/"levels"/
        "examples" keys) -- rendered as demonstration Q&A blocks prepended
        before the real query.
    :param extra_rule: optional extra rule text (e.g. from SPECIAL_RULES),
        appended after the general ANNOTATION_RULES.
    :return: The prepared prompt.
    """

    rules = ANNOTATION_RULES
    if extra_rule:
        rules = rules + extra_rule + "\n"
    rules = rules + "\n"

    prefix = ""
    if few_shot_examples:
        if list(few_shot_examples["levels"]) != list(levels):
            raise ValueError(
                f"Few-shot levels for {var_name!r} do not match the runtime levels: "
                f"{few_shot_examples['levels']} != {levels}"
            )
        prefix = "".join(
            _render_few_shot_example(
                ex["story"], few_shot_examples["var_name"], few_shot_examples["levels"], ex["answer"],
                extra_rule=extra_rule,
            )
            for ex in few_shot_examples["examples"].values()
        )

    prompt = (
        "Input: Consider the following text:\n\n"
        + text
        + "\n\n"
        + rules
        + f"Based on the text, determine the person's {var_name}. "
        + "Return exactly the single capital letter corresponding to your chosen option below. "
        + "Do not include a period, explanation, or any other text.\n"
    )
    answers, answer_mapping = prepare_answers(levels)
    prompt += answers
    prompt += "\nOutput: "
    return prefix + prompt, answer_mapping


def annotate_data(model, tokenizer, device, texts_by_attempt, var_dict, var_names, var_ord,
                  engine: str = "transformers", cache_path: str = None,
                  dataset: str = None, few_shot: bool = False):
    """
    Annotate stories against var_dict, retrying with later attempts when an
    earlier story is missing information.

    :param texts_by_attempt: list of K parallel text lists (one list per
        generation attempt, each of length nsamp, row-index-aligned). Attempt
        1 is tried first for every row; a row only advances to attempt 2, 3, ...
        if any variable came back "Answer not available" (NA) on the prior
        attempt. Rows are never reordered or dropped.
    :param dataset: dataset name, used with each var to look up few-shot
        demos in py/few_shot_examples.FEW_SHOT_EXAMPLES. Required if
        few_shot=True.
    :param few_shot: if True, prepend the hand-written demonstration examples
        for (dataset, var) to each variable's annotation prompt, when present.
    :return: DataFrame with one column per var_dict variable plus an
        "n_attempts" column (the 1-indexed attempt at which the row resolved,
        or len(texts_by_attempt) if it never did).
    """
    if engine not in {"vllm", "transformers"}:
        raise ValueError(f"Unknown engine: {engine!r}")
    if not texts_by_attempt:
        raise ValueError("texts_by_attempt must contain at least one attempt")

    n_rounds = len(texts_by_attempt)
    nsamp = len(texts_by_attempt[0])
    for attempt, texts in enumerate(texts_by_attempt, start=1):
        if len(texts) != nsamp:
            raise ValueError(
                f"Generation attempt {attempt} contains {len(texts)} rows; expected {nsamp}"
            )
    var_list = list(var_dict.keys())
    df = pd.DataFrame(index=range(nsamp), columns=var_list)
    n_attempts = pd.Series(0, index=range(nsamp), dtype=int)
    remaining = pd.Index(range(nsamp))
    cache_records = []

    if engine == "vllm":
        from vllm import SamplingParams
        vllm_tokenizer = model.get_tokenizer()
        model_id = _model_id(model=model)
        template_kwargs = _chat_template_kwargs(model_id)

        for attempt, texts in enumerate(texts_by_attempt, start=1):
            if len(remaining) == 0:
                break
            print(f"    Annotating attempt {attempt}/{n_rounds} ({len(remaining)} rows)")

            for var, levels in tqdm(var_dict.items(), desc=f"Annotating (attempt {attempt})"):

                var_name = var_names.get(var, var)
                fs_examples = FEW_SHOT_EXAMPLES.get((dataset, var)) if few_shot else None
                extra_rule = SPECIAL_RULES.get((dataset, var))
                _, answer_mapping = prepare_answers(levels)
                letters = list(answer_mapping.keys())
                letter_to_id = _single_token_answer_ids(vllm_tokenizer, letters)
                letter_ids = list(letter_to_id.values())
                id_to_letter = {tid: letter for letter, tid in letter_to_id.items()}

                sub_texts = [texts[pos] for pos in remaining]
                ann_prompts = [prep_ann_prompt(text, var_name, levels, fs_examples, extra_rule)[0] for text in sub_texts]
                chat_ann_prompts = [
                    vllm_tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=True,
                        return_dict=False,  # add
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                    for p in ann_prompts
                ]
                sp = SamplingParams(max_tokens=1, temperature=0, allowed_token_ids=letter_ids)
                vllm_inputs = [{"prompt_token_ids": p} for p in chat_ann_prompts]
                outputs = model.generate(vllm_inputs, sp)

                for pos, output, prompt in zip(remaining, outputs, ann_prompts):
                    if not output.outputs[0].token_ids:
                        raise RuntimeError("Annotator produced no token")
                    tid = output.outputs[0].token_ids[0]
                    try:
                        letter = id_to_letter[tid]
                    except KeyError as exc:
                        raise RuntimeError(f"Annotator produced unexpected token ID {tid}") from exc
                    pred_answer = answer_mapping[letter]
                    df.loc[pos, var] = pred_answer

                    if cache_path is not None:
                        cache_records.append({
                            "row": pos,
                            "variable": var,
                            "attempt": attempt,
                            "prompt": prompt,
                            "response": letter,
                        })

            n_attempts.loc[remaining] = attempt
            still_invalid = df.loc[remaining, var_list].isna().any(axis=1)
            remaining = still_invalid[still_invalid].index

    else:  # transformers
        if tokenizer is None:
            raise ValueError("tokenizer is required when engine='transformers'")
        model_id = _model_id(model=model, tokenizer=tokenizer)
        template_kwargs = _chat_template_kwargs(model_id)

        for attempt, texts in enumerate(texts_by_attempt, start=1):
            if len(remaining) == 0:
                break
            print(f"    Annotating attempt {attempt}/{n_rounds} ({len(remaining)} rows)")

            for var, levels in tqdm(var_dict.items(), desc=f"Annotating (attempt {attempt})"):

                _, answer_mapping = prepare_answers(levels)
                letter_to_id = _single_token_answer_ids(tokenizer, answer_mapping.keys())
                var_name = var_names.get(var, var)
                fs_examples = FEW_SHOT_EXAMPLES.get((dataset, var)) if few_shot else None
                extra_rule = SPECIAL_RULES.get((dataset, var))

                for pos in remaining:
                    text = texts[pos]
                    ann_prompt, _ = prep_ann_prompt(text, var_name, levels, fs_examples, extra_rule)
                    inputs = _apply_transformers_chat_template(
                        tokenizer,
                        [ann_prompt],
                        template_kwargs,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs).logits
                        next_token_logits = outputs[:, -1, :]

                    letters = list(answer_mapping.keys())
                    label_logits = torch.stack(
                        [next_token_logits[0, letter_to_id[letter]] for letter in letters]
                    )
                    pred_letter = letters[int(torch.argmax(label_logits).item())]
                    pred_answer = answer_mapping[pred_letter]
                    df.loc[pos, var] = pred_answer

                    if cache_path is not None:
                        cache_records.append({
                            "row": pos,
                            "variable": var,
                            "attempt": attempt,
                            "prompt": ann_prompt,
                            "response": pred_letter,
                        })

            n_attempts.loc[remaining] = attempt
            still_invalid = df.loc[remaining, var_list].isna().any(axis=1)
            remaining = still_invalid[still_invalid].index

    for var, levels in var_dict.items():
        df[var] = pd.Categorical(df[var], categories=levels, ordered=var_ord[var])
    df["n_attempts"] = n_attempts

    if cache_path is not None:
        _save_parquet(cache_records, cache_path)

    return df


# enforce levels helper
def enforce_levels(df, var_dict, var_ord):
    """Ensure all columns have the same categorical levels as the source data."""
    for var, levels in var_dict.items():
        if var in df.columns:
            df[var] = pd.Categorical(df[var], categories=levels, ordered=var_ord[var])
    return df


# pre-processing function to remove categoricals
def clean_cats(df, X):

    # binary categoricals: map yes/no to 1/0
    for col in df.columns:
        dt = df[col].dtype
        if not isinstance(dt, pd.CategoricalDtype):
            continue
        cats = list(df[col].cat.categories)
        if len(cats) == 2:
            if set(cats) == {"yes", "no"}:
                df[col] = df[col].map({"yes": 1, "no": 0})
            else:
                df[col] = df[col].cat.codes  # map to 0/1

    # multi-level categoricals
    for col in list(df.columns):  # df may change when we add dummies
        dt = df[col].dtype
        if not isinstance(dt, pd.CategoricalDtype):
            continue

        s = df[col]
        cats = list(s.cat.categories)
        if len(cats) <= 2:
            continue

        if s.cat.ordered:
            # ordered -> integer codes (1..K)
            df[col] = s.cat.codes + 1
        else:
            # unordered, non-binary, and not in X -> one-hot
            if col in X:
                continue
            dummies = pd.get_dummies(s, prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df
