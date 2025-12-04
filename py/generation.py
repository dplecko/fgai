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
def varlist_to_prompt(var_dict, var_names):

    prompt_start = (
        "You are a data generator. Follow the rules strictly.\n"
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


@torch.inference_mode()
def gen_data_batched(
    df=None,
    var_dict=None,
    var_names=None,
    nsamp: int = 100,
    model=None,
    tokenizer=None,
    device=None,
    max_new_tokens: int = 256,
    temperature: float = 1,
    top_p: float = 0.9,
    batch_size: int = 16,  # tune per GPU memory
):

    # 0) check if df is given; if so, build prompts from it
    if df is not None:
        nsamp = df.shape[0]
        known_vars = df.columns.tolist()
        # construct the var_dict; known values are replaced by the df values and prompt is constructed per row
        prompts = []
        for i in range(nsamp):
            dyn_dict = var_dict.copy()
            for var in known_vars:
                dyn_dict[var] = [str(df.loc[i, var])]
            prompt_i = varlist_to_prompt(dyn_dict, var_names)
            prompts.append(prompt_i)
    else:
        prompt = varlist_to_prompt(var_dict, var_names)
        prompts = [prompt for _ in range(nsamp)]

    # 1) Build all prompts up front (vectorized)
    out_texts = []

    # 2) Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating (batched)"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize with padding; keep per-sample lengths for slicing
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_lens = attention_mask.sum(dim=1)  # effective prompt lengths per sample

        # Generate
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,  # NOTE: what is this for?
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(
                tokenizer,
                "pad_token_id",
                getattr(tokenizer, "eos_token_id", None),
            ),
            use_cache=True,
        )

        # 3) Slice out continuations per-sample and decode
        for j in range(gen.size(0)):
            ilen = int(input_lens[j].item())
            seq = gen[j, ilen:] if gen.size(1) > ilen else gen[j]
            raw = tokenizer.decode(seq, skip_special_tokens=True).strip()

            # check if the raw output starts with a proper <story> tag
            story = re.sub(r"<[^>]+>|{[^}]+}", "", raw)

            # hard cap ~200 words as guard
            words = story.split()
            if len(words) > 200:
                story = " ".join(words[:200]).rstrip()

            out_texts.append(story)

        # free some memory sooner
        del enc, input_ids, attention_mask, gen
        torch.cuda.empty_cache()

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

    # make this conditional with try and except with a breakpoint() if it fails
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


def annotate_data(model, tokenizer, device, texts, var_dict, var_ord):

    # initiate a data.frame to hold the results
    df = pd.DataFrame(columns=var_dict.keys())

    # for loop over all texts
    for var, levels in tqdm(var_dict.items(), desc="Annotating data"):

        _, answer_mapping = prepare_answers(levels)

        for i, text in enumerate(texts):
            # extract relevant variables from var_dict

            inputs = tokenizer(
                prep_ann_prompt(text, var, levels)[0], return_tensors="pt"
            ).to(device)
            level_ids = [
                [tokenizer.convert_tokens_to_ids(tok) for tok in ans]
                for ans in answer_mapping.keys()
            ]

            with torch.no_grad():
                outputs = model(**inputs).logits
                next_token_logits = outputs[
                    :, -1, :
                ]  # Last token in the input sequence
                probs = torch.softmax(next_token_logits, dim=-1)

            # Normalise probability mass over the provided answers
            level_probs = [
                sum(probs[0, tid].item() for tid in ids) for ids in level_ids
            ]

            # get the predicted answer from the list
            pred_idx = max(range(len(level_probs)), key=level_probs.__getitem__)
            pred_answer = answer_mapping[list(answer_mapping.keys())[pred_idx]]

            # save the answer to the data frame for the i-th
            df.loc[i, var] = pred_answer

        # ensure that the variable is categorical with correct levels
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
