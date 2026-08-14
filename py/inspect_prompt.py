# One-off inspection tool: pull a real story out of an actual generation
# cache and run it through the real prep_ann_prompt (imported, not
# reimplemented) so the rendered prompt can be eyeballed for correctness --
# no hand-picked/curated example, first matching file and first row.
#
# Usage (inside the fgai container):
#   python3 -m py.inspect_prompt --dataset census_income --var education --few_shot
#   python3 -m py.inspect_prompt --dataset nsduh --var race
# Or interactively (e.g. Positron/IPython console, where sys.argv is the
# kernel's own args, not yours):
#   from py.inspect_prompt import main
#   main(["--dataset", "census_income", "--var", "education", "--few_shot"])
import argparse
import glob
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from py.generation import prep_ann_prompt, SPECIAL_RULES
from py.few_shot_examples import FEW_SHOT_EXAMPLES
from py.data_helpers import load_data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--var", type=str, required=True)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--row", type=int, default=0, help="row index within the first matching gen file")
    args = parser.parse_args(argv)

    matches = sorted(glob.glob(f"data/cache/{args.dataset}_*_gen.parquet"))
    matches = [m for m in matches if "_attempt" not in m]
    if not matches:
        raise FileNotFoundError(f"no data/cache/{args.dataset}_*_gen.parquet files found")
    gen_file = matches[0]
    print(f"Using gen cache: {gen_file}")

    gen_df = pd.read_parquet(gen_file)
    story = gen_df["response"].iloc[args.row]
    print(f"Row {args.row} story:\n{story}\n")
    print("=" * 80)

    _df, var_dict, var_names, _var_ord, _sfm, _context = load_data(args.dataset)
    if args.var not in var_dict:
        raise KeyError(f"{args.var!r} not in var_dict for {args.dataset}; options: {list(var_dict.keys())}")
    levels = var_dict[args.var]
    var_name = var_names.get(args.var, args.var)
    extra_rule = SPECIAL_RULES.get((args.dataset, args.var))

    fs_examples = FEW_SHOT_EXAMPLES.get((args.dataset, args.var)) if args.few_shot else None
    if args.few_shot and fs_examples is None:
        print(f"[warn] --few_shot given but no FEW_SHOT_EXAMPLES entry for ({args.dataset}, {args.var}) -- zero-shot")

    prompt, answer_mapping = prep_ann_prompt(story, var_name, levels, fs_examples, extra_rule)

    print(f"PROMPT (var={args.var}, few_shot={bool(fs_examples)}, extra_rule={'yes' if extra_rule else 'no'}):")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print(f"answer_mapping: {answer_mapping}")


if __name__ == "__main__":
    main()
