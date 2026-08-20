
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

N_SAMPLES <- 140
OUT <- "results/manual_labels.csv"
dataset <- "nsduh"
ann_mod <- "llama3_70b"
GENERATORS <- c("llama3_8b", "llama3_70b", "qwen35_9b", "qwen35_27b",
                "gemma3_4b", "gemma3_27b", "deepseek_7b", "deepseek_r1",
                "ministral3_8b", "phi4")
N_PER_MODEL <- ceiling(N_SAMPLES / length(GENERATORS))

set.seed(42)
all_pool <- rbindlist(lapply(GENERATORS, function(g) {
  d <- as.data.table(read_parquet(f("data/cache/{dataset}_{g}_{ann_mod}__ann.parquet")))
  d[, `:=`(row_id = .I, gen_model = g)]
  d <- d[, .SD[sample(.N)], by = variable]
  d <- d[, .SD[seq_len(min(.N, ceiling(N_PER_MODEL / uniqueN(variable))))], by = variable]
  head(d[sample(.N)], N_PER_MODEL)
}))
all_pool <- all_pool[sample(.N)]
all_pool <- head(all_pool, N_SAMPLES)

done <- if (file.exists(OUT)) fread(OUT) else
  data.table(gen_model = character())

# top up to N_PER_MODEL per generator
counts <- done[, .N, by = gen_model]
todo <- all_pool[, {
  have <- counts[gen_model == .BY$gen_model, N]
  have <- if (length(have)) have else 0
  need <- max(0, N_PER_MODEL - have)
  head(.SD, need)
}, by = gen_model]

cat("Labeling", nrow(todo), "samples (already done:", nrow(done), ")\n")

for (i in seq_len(nrow(todo))) {
  r <- todo[i]
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("Sample", nrow(done) + i, "/", N_SAMPLES,
      " | gen:", r$gen_model, " | variable:", r$variable, " | LLM:", r$response, "\n")
  cat(strrep("-", 70), "\n", sep = "")
  cat(r$prompt, "\n")
  cat(strrep("-", 70), "\n", sep = "")
  
  ans <- toupper(trimws(readline("Your label ('s'=NA, 'q'=quit): ")))
  if (ans == "Q") break
  if (ans == "S") ans <- NA_character_
  
  fwrite(
    data.table(row_id = r$row_id, variable = r$variable,
               llm = r$response, human = ans, gen_model = r$gen_model),
    OUT, append = file.exists(OUT)
  )
}

cat("\nDone. Saved to", OUT, "\n")

hitl <- fread(OUT)
with_seed(123, hitl <- hitl[, .SD[sample(.N, min(.N, 15))], by = gen_model])
mean(hitl$human == "")
mean(hitl[human != ""]$llm == hitl[human != ""]$human, na.rm = TRUE)


