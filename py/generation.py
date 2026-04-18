import os
import re
import string
import torch
from tqdm import tqdm
import pandas as pd


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
def varlist_to_prompt(var_dict, var_names, context=None):

    ctx_line = f"CONTEXT: {context}\n" if context else ""
    prompt_start = (
        "You are a data generator. Follow the rules strictly.\n"
        + ctx_line +
        "RULES:\n"
        "1) Write a single narrative enclosed in <story>...</story>.\n"
        "2) Do NOT include headings, lists, analysis, or any text outside the tags.\n"
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

    prompt_end = "OUTPUT FORMAT:\n" "<story>\n" "(your narrative here)\n" "</story>\n"

    prompt = prompt_start + "\n" + known + "\n" + unknown + "\n\n" + prompt_end
    return prompt


def extract_tag(text: str, tag: str = "story") -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: if the model disobeys, strip known headings and return the first paragraph
    text = re.sub(r"(?is)^#+.*?$", "", text)  # markdown headers
    text = re.sub(r"(?is)^(analyzing errors|analysis).*?$", "", text).strip()
    # take first non-empty paragraph
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if p:
            return p
    return text.strip()


def _save_parquet(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
            prompts.append(varlist_to_prompt(dyn_dict, var_names, context=context))
    else:
        prompt = varlist_to_prompt(var_dict, var_names, context=context)
        prompts = [prompt for _ in range(nsamp)]

    out_texts = []

    if engine == "vllm":
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = model.generate(prompts, sp)
        for output in outputs:
            raw = output.outputs[0].text.strip()
            story = re.sub(r"<[^>]+>|{[^}]+}", "", raw)
            words = story.split()
            if len(words) > 200:
                story = " ".join(words[:200]).rstrip()
            out_texts.append(story)

    else:  # transformers
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating (batched)"):
            batch_prompts = prompts[i : i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            input_lens = attention_mask.sum(dim=1)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
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

            for j in range(gen.size(0)):
                ilen = int(input_lens[j].item())
                seq = gen[j, ilen:] if gen.size(1) > ilen else gen[j]
                raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
                story = re.sub(r"<[^>]+>|{[^}]+}", "", raw)
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
def prepare_answers(levels):
    """
    Prepare the answers for the model.
    :param levels: A list of possible answers.
    :return: A list of tokenized answers.
    """
    if len(levels) > 26:
        raise ValueError("Supports up to 26 items (A-Z)")

    letters = string.ascii_uppercase  # 'A', 'B', ...
    mapping = {letters[i]: item for i, item in enumerate(levels)}
    answer_key = "\n".join(f"{k}. {v}" for k, v in mapping.items())

    return answer_key, mapping


def prep_ann_prompt(text, var_name, levels):
    """
    Prepare the prompt for the model.
    :param prompt: The initial prompt.
    :param levels: A list of possible answers.
    :return: The prepared prompt.
    """

    # clean from any appearance of "{" or "}" which may break format
    text = re.sub(r"{|}", "", text)

    try:
        prompt = (
            "Input: Based on the following text:\n\n"
            + text
            + "\n\ndetermine the person's {}. "
            + "Begin your answer with the capital letter corresponding to your chosen option below, followed by a period.\n"
        ).format(var_name)
    except Exception as e:
        breakpoint()
    answers, answer_mapping = prepare_answers(levels)
    prompt += answers
    prompt += "\nOutput: "
    return prompt, answer_mapping


def annotate_data(model, tokenizer, device, texts, var_dict, var_names, var_ord,
                  engine: str = "transformers", cache_path: str = None):

    df = pd.DataFrame(columns=var_dict.keys())
    cache_records = []

    if engine == "vllm":
        from vllm import SamplingParams
        vllm_tokenizer = model.get_tokenizer()

        for var, levels in tqdm(var_dict.items(), desc="Annotating data"):
            var_name = var_names.get(var, var)
            _, answer_mapping = prepare_answers(levels)
            letters = list(answer_mapping.keys())
            letter_ids = [vllm_tokenizer.convert_tokens_to_ids(l) for l in letters]

            ann_prompts = [prep_ann_prompt(text, var_name, levels)[0] for text in texts]
            breakpoint()
            sp = SamplingParams(max_tokens=1, temperature=0, allowed_token_ids=letter_ids)
            outputs = model.generate(ann_prompts, sp)

            for i, output in enumerate(outputs):
                letter = output.outputs[0].text.strip().upper()
                pred_answer = answer_mapping.get(letter, levels[0])
                df.loc[i, var] = pred_answer

                if cache_path is not None:
                    cache_records.append({
                        "variable": var,
                        "prompt": ann_prompts[i],
                        "response": letter,
                    })

            df[var] = pd.Categorical(df[var], categories=levels, ordered=var_ord[var])

    else:  # transformers
        for var, levels in tqdm(var_dict.items(), desc="Annotating data"):

            _, answer_mapping = prepare_answers(levels)

            for i, text in enumerate(texts):
                var_name = var_names.get(var, var)
                ann_prompt, _ = prep_ann_prompt(text, var_name, levels)
                inputs = tokenizer(ann_prompt, return_tensors="pt").to(device)
                level_ids = [
                    [tokenizer.convert_tokens_to_ids(tok) for tok in ans]
                    for ans in answer_mapping.keys()
                ]

                with torch.no_grad():
                    outputs = model(**inputs).logits
                    next_token_logits = outputs[:, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)

                level_probs = [
                    sum(probs[0, tid].item() for tid in ids) for ids in level_ids
                ]

                pred_idx = max(range(len(level_probs)), key=level_probs.__getitem__)
                pred_answer = answer_mapping[list(answer_mapping.keys())[pred_idx]]
                df.loc[i, var] = pred_answer

                if cache_path is not None:
                    cache_records.append({
                        "variable": var,
                        "prompt": ann_prompt,
                        "response": list(answer_mapping.keys())[pred_idx],
                    })

            df[var] = pd.Categorical(df[var], categories=levels, ordered=var_ord[var])

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
