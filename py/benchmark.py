# python -m py.benchmark --model llama3_8b --engine transformers
# python -m py.benchmark --model deepseek_r1 --engine vllm
import argparse
import time
import torch
from py.model_load import MODEL_PATHS, get_model, get_vllm_model, model_batchsize
from py.generation import gen_data_batched, annotate_data

NSAMP = 16

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3_8b")
parser.add_argument("--engine", type=str, default="transformers",
                    choices=["transformers", "vllm"])
args = parser.parse_args()

model_name = args.model
engine     = args.engine
model_path = MODEL_PATHS[model_name]

# var_dict split into known (gen inputs) and unknown (to be annotated)
known_var_dict = {
    "race": ["white"],
}
unknown_var_dict = {
    "age":    ["young adult", "middle-aged", "senior"],
    "sex":    ["male", "female"],
    "income": ["low", "medium", "high"],
}
var_dict  = {**known_var_dict, **unknown_var_dict}
var_names = {k: k for k in var_dict}
var_ord   = {k: False for k in unknown_var_dict}

print(f"Benchmark: model={model_name}, engine={engine}, nsamp={NSAMP}")
print(f"  Variables to generate: {list(var_dict.keys())}")
print(f"  Variables to annotate: {list(unknown_var_dict.keys())}")
print("Loading model...")

if engine == "transformers":
    model, tokenizer, device = get_model(model_path)
    batch_size = model_batchsize(model_name)
else:
    model = get_vllm_model(model_path)
    tokenizer = device = None
    batch_size = 1  # ignored by vllm engine

print("Model loaded.\n")

# ── Generation ───────────────────────────────────────────────────────────────
print(f"[1/2] Generating {NSAMP} stories...")
t0 = time.perf_counter()

texts = gen_data_batched(
    df=None,
    var_dict=var_dict,
    var_names=var_names,
    context=None,
    nsamp=NSAMP,
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=256,
    batch_size=batch_size,
    engine=engine,
)

t_gen = time.perf_counter() - t0
print(f"    done in {t_gen:.1f}s  ({NSAMP/t_gen:.2f} samples/s)\n")

# ── Annotation ───────────────────────────────────────────────────────────────
print(f"[2/2] Annotating {NSAMP} stories × {len(unknown_var_dict)} variables...")
t1 = time.perf_counter()

annotate_data(
    model, tokenizer, device,
    texts, unknown_var_dict, var_ord,
    engine=engine,
)

t_ann = time.perf_counter() - t1
t_total = t_gen + t_ann
print(f"    done in {t_ann:.1f}s  ({NSAMP/t_ann:.2f} samples/s)\n")

# ── Summary ──────────────────────────────────────────────────────────────────
print("─" * 40)
print(f"Engine:      {engine}")
print(f"Model:       {model_name}")
print(f"Samples:     {NSAMP}")
print(f"Generation:  {t_gen:.1f}s  ({1000*t_gen/NSAMP:.0f} ms/sample)")
print(f"Annotation:  {t_ann:.1f}s  ({1000*t_ann/NSAMP:.0f} ms/sample)")
print(f"Total:       {t_total:.1f}s  ({1000*t_total/NSAMP:.0f} ms/sample)")
