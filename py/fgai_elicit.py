# nohup python workspace/fgai_elicit.py > fgai_elicit.log 2>&1 &
from py.common import *
from py.model_load import MODEL_PATHS
from py.fgai_helpers import *

dataset = "nsduh"
df = pd.read_parquet(f"data/raw/{dataset}.parquet")
var_names = get_var_names(dataset)
var_dict = {}
for var in var_names.keys():
    # extract the variable categories
    var_dict[var] = sorted(df[var].unique())

model_path = MODEL_PATHS["llama3_8b_instruct"][0]
model, tokenizer, device = get_model(model_path, prefer_gpu_idx=0)

nsamp = 10000

texts = gen_data_batched(
    nsamp, model, tokenizer, device, varlist_to_prompt(var_dict, var_names)
)

# save the intermediate texts list to a pickle file
import pickle

with open("data/fgai/{dataset}_fgai_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

# read the pickle file
with open("data/fgai/{dataset}_fgai_texts.pkl", "rb") as f:
    texts = pickle.load(f)

df_m = annotate_data(model, tokenizer, device, texts, var_dict)

# go over all columns in df, and if they are categorical, make df_m inherit their category and order
for col in df.columns:
    if col in df_m.columns:
        dt = df[col].dtype
        if isinstance(dt, pd.CategoricalDtype):
            df_m[col] = pd.Categorical(
                df_m[col], categories=df[col].cat.categories, ordered=dt.ordered
            )

# sample n samp rows from original data, with weights
df_w = (
    df[vars].sample(n=nsamp, weights=df["weight"], replace=False).reset_index(drop=True)
)

# rbind the two dataframes, and add a binary 0/1 env column
df_m["env"] = 1
df_w["env"] = 0
df_cmb = pd.concat([df_m, df_w], ignore_index=True)


df_res = clean_cats(df_cmb, X=["race"])
df_res.to_parquet("data/fgai/{dataset}_envs.parquet", index=False)

# auditing the model labels
import pickle

with open("data/fgai/nsduh_fgai_texts.pkl", "rb") as f:
    texts = pickle.load(f)

df_m = pd.read_parquet("data/fgai/nsduh_envs.parquet")
df_m = df_m[df_m["env"] == 1].reset_index(drop=True)

# sample 10 random rows to inspect
import random

random.seed(42)
sample_indices = random.sample(range(len(texts)), 10)

sample_indices = [1824, 409, 4506, 4012, 3657, 2286, 1679, 8935, 1424, 9674]

print(texts[sample_indices[3]])

idx = 9674
print(texts[idx])
# need to convert the levels back to original labels
for var in df_m.columns:
    if var in ["age", "edu", "income"]:
        print(var, df[var].cat.categories[df_m.loc[idx, var] - 1])
    elif var in ["sex"]:
        print(var, df[var].cat.categories[df_m.loc[idx, var]])
    else:
        print(var, df_m.loc[idx, var])

# automated checking


# load the data from fgai/data/story-truth.csv
df_truths = pd.read_csv("../fgai/data/story-truth.csv")

# verify that each column value is in the correct levels
for var in df_truths.columns:

    if var not in var_dict:
        continue

    # correct levels
    levels = var_dict[var]

    # check if all values in df_truths[var] are in levels
    invalid_values = set(df_truths[var].unique()) - set(levels)
    if len(invalid_values) > 0:
        print(f"Variable {var} has invalid values: {invalid_values}")

# get the automatic check
for i in range(len(df_truths)):

    idx = df_truths.loc[i, "index"]

    for var in df_truths.columns:
        if var not in var_dict:
            continue

        true_value = df_truths.loc[i, var]

        if var in ["age", "edu", "income"]:
            pred_value = df[var].cat.categories[df_m.loc[idx, var] - 1]
        elif var in ["sex", "alc_monthly", "cig_monthly", "mj_monthly", "coc_ever"]:
            pred_value = df[var].cat.categories[df_m.loc[idx, var]]
        else:
            pred_value = df_m.loc[idx, var]

        if true_value != pred_value:
            print(
                f"Row {i}, Variable {var}: true value = {true_value}, predicted value = {pred_value}"
            )
