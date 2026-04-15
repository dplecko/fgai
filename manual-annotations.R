
# looking at annotations manually
dran <- read_parquet("data/cache/nsduh_llama3_70b_llama3_70b__ann.parquet")
dr1 <- read_parquet("data/cache/nsduh_llama3_70b__gen.parquet")

read_parquet("data/nsduh_deepseek_r1_.parquet")

cat(dr1[1, ][["prompt"]])


cat(dr1[1, ][["response"]])