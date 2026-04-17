


dr1 <- read_parquet("data/cache/nsduh_llama3_70b_XZ_gen.parquet")

cat(dr1[100, ][["prompt"]])

cat(dr1[100, ][["response"]])

# looking at annotations manually
dran <- read_parquet("data/cache/nsduh_llama3_70b_llama3_70b_XZ_ann.parquet")

cat(dran[100, ][["prompt"]])

cat(dran[100, ][["response"]])
