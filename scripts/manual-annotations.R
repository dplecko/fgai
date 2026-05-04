

read_parquet("data/raw/brfss.parquet")$race

read_parquet("data/nsduh_llama3_70b_XZW.parquet")$age_group

dr1 <- read_parquet("data/cache/nsduh_ministral3_8b__gen.parquet")

cat(dr1[3, ][["prompt"]])

cat(dr1[3, ][["response"]])

# looking at annotations manually
dran <- read_parquet("data/cache/nsduh_ministral3_8b_ministral3_8b__ann.parquet")


dran <- read_parquet("data/cache/nsduh_llama3_8b_llama3_8b__ann.parquet")

cat(dran[2, ][["prompt"]])

cat(dran[3, ][["response"]])


cat(dran[dran$variable == "race", ]$prompt[1815])

library(data.table)
tbl <- rbindlist(lapply(DATASETS, function(ds) {
  rbindlist(lapply(MODELS, function(m) {
    df <- read_parquet(f("data/{ds}_{m}_.parquet"))
    data.table(dataset = ds, model = m,
               n_x0 = sum(df$race %in% c("White")),  # adjust per dataset
               n_x1 = sum(df$race %in% c("Black", "African American", "Hispanic")))
  }))
}))
tbl[, ratio := n_x1 / (n_x0 + n_x1)]
tbl[, n_eff := pmin(n_x1, n_x0)]
tbl[order(ratio)]
