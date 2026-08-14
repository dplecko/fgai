"""
Shared helpers for the structured (single-call JSON) annotation prompt: the
model returns target_person -> explanation -> answer in one response, in
that order, so it reasons about who the story is about and why before
committing to a letter.

These are used by generation.py's prep_ann_prompt (the production path) and
by py/goldtest_structured.py (the benchmark harness these prompts were
validated in) so the two can't silently drift apart -- prior to this file
they each had their own copy of the rules/demo text, which is exactly the
kind of duplication that lets a fix land in one place and not the other.

Prefix caching: prep_ann_prompt places the story *last* in the prompt (after
rules/demos/question/answer options/instructions) specifically so that
entire preamble is byte-identical across every row for a given (dataset,
variable) pair, letting vLLM's automatic prefix caching (on by default --
https://docs.vllm.ai/en/stable/design/prefix_caching/) reuse the KV cache
for it instead of recomputing the whole prompt on every row. The helpers
here (demo rendering) only produce the preamble content; the story-last
ordering itself lives in prep_ann_prompt.
"""
import json
import re

# Order the demos are shown in. "special_rule" is only present for the 4
# (dataset, variable) pairs that carry a SPECIAL_RULES entry -- render_demos_
# section skips any name not present in a given pair's "examples" dict.
DEMO_ORDER = ["first_person", "rewrite", "contradiction", "unclear", "clear", "special_rule"]

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_RE = re.compile(r'"answer"\s*:\s*"?([A-Za-z])')

# Demo stories are all written "Name, ..." / "Name is ..." / "Name's ..." or,
# for the handful of pronoun-free "unclear" demos, "The <noun> ...". Good
# enough to recover a target-person string for rendering the worked
# example's JSON answer -- this is just a rendering utility, not authored
# content.
_TARGET_RE = re.compile(r"^(The\s+\w+|[A-Z][a-zA-Z]+)")


def infer_target_person(story):
    m = _TARGET_RE.match(story)
    return m.group(1) if m else "the person"


def demo_answer_letter(answer_text, mapping):
    """mapping is prepare_answers()'s letter -> value dict (None for NA)."""
    for letter, val in mapping.items():
        if answer_text == "NA":
            if val is None:
                return letter
        elif val == answer_text:
            return letter
    raise ValueError(f"demo answer {answer_text!r} not found among mapping values {list(mapping.values())}")


def render_demo(idx, ex, var_name, answer_key, mapping):
    letter = demo_answer_letter(ex["answer"], mapping)
    target_person = infer_target_person(ex["story"])
    demo_json = json.dumps({
        "target_person": target_person,
        "explanation": ex["explanation"],
        "answer": letter,
    })
    return (
        f"Example {idx}:\n"
        f"<story>\n{ex['story']}\n</story>\n\n"
        f"Question: determine the person's {var_name}.\n\n"
        f"{answer_key}\n\n"
        f"{demo_json}\n"
    )


def render_demos_section(examples, var_name, answer_key, mapping):
    """examples: the "examples" dict of a FEW_SHOT_EXAMPLES[(dataset, var)]
    entry. Raises ValueError (via demo_answer_letter) if a demo's answer
    doesn't match any level in `mapping` -- callers should dry-run this once
    per (dataset, variable) and fall back to zero-shot on mismatch, since
    that means few_shot_examples.py's stored levels drifted from the live
    var_dict."""
    demo_blocks = [
        render_demo(i, examples[name], var_name, answer_key, mapping)
        for i, name in enumerate(DEMO_ORDER, start=1)
        if name in examples
    ]
    return (
        "Here are some example stories with correct answers, followed by a new "
        "story for you to classify:\n\n" + "\n".join(demo_blocks) +
        "\n--- Now classify this story ---\n\n"
    )


def parse_json_answer(text):
    """Parse a single-line JSON {"target_person":..., "explanation":...,
    "answer": "<letter>"} response, tolerant of the model adding stray text
    around it (parsed from free text with a forgiving fallback, not enforced
    via guided decoding). Returns (letter_or_None, target_person_or_None,
    explanation_or_None)."""
    target_person = explanation = None
    m = JSON_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            target_person = parsed.get("target_person")
            explanation = parsed.get("explanation")
            letter = str(parsed.get("answer", "")).strip().upper()
            if len(letter) == 1:
                return letter, target_person, explanation
        except (json.JSONDecodeError, AttributeError):
            pass
    m = ANSWER_RE.search(text)
    letter = m.group(1).upper() if m else None
    return letter, target_person, explanation
