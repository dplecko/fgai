# nohup python workspace/elicit.py --model llama3_8b --batch 1 > logs/elicit_llama3_8b.log 2>&1 &
import argparse
import torch
from py.common import *
from py.model_load import MODEL_PATHS, get_model, get_vllm_model, model_batchsize
from py.generation import *
from py.data_helpers import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3_8b")
parser.add_argument("--ann_model", type=str, default=None,
                    help="Annotator model name (defaults to --model)")
parser.add_argument("--engine", type=str, default="vllm",
                    choices=["vllm", "transformers"])
parser.add_argument(
    "--batch", type=int, default=1, help="1 = rows 0:8192, 2 = rows 8192:16384"
)
args = parser.parse_args()

# --- settable directly for interactive use ---
model_name     = args.model           # e.g. model_name = "llama3_8b"
ann_model_name = args.ann_model or model_name  # e.g. ann_model_name = "llama3_8b"
engine         = args.engine          # e.g. engine = "vllm"
batch_num      = args.batch
# --------------------------------------------

BATCH_SIZE = 8192
same_model = (ann_model_name == model_name)
ann_only = False

datasets = [
    "nsduh",
    "brfss",
    "census_income",
]

def main():
    # ── preload all dataset info ────────────────────────────────────────────────
    dataset_info = {}
    for dataset in datasets:
        df, var_dict, var_names, var_ord, sfm, context = load_data(dataset)
        df = df.sample(
            n=2 * BATCH_SIZE, weights=df["weight"], replace=True, random_state=0
        ).reset_index(drop=True)
        start = (batch_num - 1) * BATCH_SIZE
        df = df.iloc[start : start + BATCH_SIZE].reset_index(drop=True)
        var_groups = {
            "": [],
            "XZ": sfm["X"] + sfm["Z"],
            "XZW": sfm["X"] + sfm["Z"] + sfm["W"],
            "XZWY": sfm["X"] + sfm["Z"] + sfm["W"] + sfm["Y"],
        }
        dataset_info[dataset] = dict(
            df=df, var_dict=var_dict, var_names=var_names, var_ord=var_ord,
            sfm=sfm, context=context, var_groups=var_groups,
        )

    torch.manual_seed(2025 + batch_num)


    # ── PHASE 1: generation ─────────────────────────────────────────────────────
    def _load_model(name):
        path = MODEL_PATHS[name]
        if engine == "transformers":
            return get_model(path)          # (model, tokenizer, device)
        else:
            return get_vllm_model(path), None, None


    def _unload_model(model):
        del model
        if engine == "transformers":
            torch.cuda.empty_cache()


    print("=== Phase 1: Generation ===")
    if not ann_only:
        gen_model, gen_tokenizer, gen_device = _load_model(model_name)

    generated = {}  # (dataset, group_name) -> texts

    for dataset in datasets:
        info = dataset_info[dataset]
        df, var_dict, var_names = info["df"], info["var_dict"], info["var_names"]
        context, var_groups = info["context"], info["var_groups"]
        nsamp = len(df)
        print(f"\n  {dataset} (batch {batch_num})")

        for group_name, vars in var_groups.items():
            if group_name == "XZWY":
                continue  # no generation needed

            df_sub = df[vars] if group_name != "" else None
            gen_cache = f"data/cache/{dataset}_{model_name}_{group_name}_gen.parquet"

            # skip generation if this batch's texts are already cached
            texts = None
            if os.path.exists(gen_cache):
                cached = pd.read_parquet(gen_cache)
                if len(cached) >= batch_num * BATCH_SIZE:
                    start = (batch_num - 1) * BATCH_SIZE
                    texts = cached["response"].iloc[start : start + BATCH_SIZE].tolist()
                    print(f"    [{group_name}] loaded {len(texts)} texts from cache, skipping generation")

            if texts is None:
                # save previous batches before gen_data_batched overwrites the cache
                prev_cache = pd.read_parquet(gen_cache) if os.path.exists(gen_cache) else None
                texts = gen_data_batched(
                    df_sub, var_dict, var_names, context, nsamp,
                    gen_model, gen_tokenizer, gen_device,
                    batch_size=model_batchsize(model_name) if engine == "transformers" else 1,
                    engine=engine,
                    cache_path=gen_cache,
                )
                # accumulate cache across batches
                if prev_cache is not None:
                    pd.concat([prev_cache, pd.read_parquet(gen_cache)], ignore_index=True).to_parquet(gen_cache, index=False)

            generated[(dataset, group_name)] = texts

    # unload gen model before loading ann model (if they differ)
    if not same_model and not ann_only:
        print("\nUnloading generation model from VRAM...")
        _unload_model(gen_model)
        del gen_model


    # ── PHASE 2: annotation + save ──────────────────────────────────────────────
    print("\n=== Phase 2: Annotation ===")

    # Phase 2 model load
    if same_model and not ann_only:
        ann_model, ann_tokenizer, ann_device = gen_model, gen_tokenizer, gen_device
    else:
        ann_model, ann_tokenizer, ann_device = _load_model(ann_model_name)

    for dataset in datasets:
        info = dataset_info[dataset]
        df, var_dict, var_ord = info["df"], info["var_dict"], info["var_ord"]
        var_groups = info["var_groups"]
        nsamp = len(df)
        print(f"\n  {dataset} (batch {batch_num})")

        for group_name, vars in var_groups.items():
            out_path  = f"data/{dataset}_{model_name}_{group_name}.parquet"
            ann_cache = f"data/cache/{dataset}_{model_name}_{ann_model_name}_{group_name}_ann.parquet"

            if group_name == "XZWY":
                df_new = enforce_levels(df[vars].copy(), var_dict, var_ord)
            else:
                df_sub = df[vars] if group_name != "" else None
                texts  = generated[(dataset, group_name)]
                var_dict_sub = {k: v for k, v in var_dict.items() if k not in vars}
                var_names_sub = {k: v for k, v in var_names.items() if k not in vars}

                df_ann = annotate_data(
                    ann_model, ann_tokenizer, ann_device,
                    texts, var_dict_sub, var_names_sub, var_ord,
                    engine=engine,
                    cache_path=ann_cache,
                )
                df_new = (
                    df_ann if group_name == ""
                    else pd.concat(
                        [df_sub.reset_index(drop=True), df_ann.reset_index(drop=True)],
                        axis=1,
                    )
                )
                df_new = enforce_levels(df_new, var_dict, var_ord)

            if batch_num > 1 and os.path.exists(out_path):
                df_prev = pd.read_parquet(out_path)
                df_new = pd.concat([df_prev, df_new], ignore_index=True)

            if not ann_only:
                df_new.to_parquet(out_path, index=False)
                print(f"    Saved {out_path} ({len(df_new)} rows)")

if __name__ == "__main__":
    main()