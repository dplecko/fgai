
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- qwen25_72b / commandrp_104b vs. human labels, reusing the existing ----
# --- 214-row validation sample (results/manual_labels.csv). That file was --
# --- built (annotator-validation.R) against the OLD llama3_70b _ann logs, --
# --- on nsduh only. Read-only throughout: no cache files are touched. -----
#
# row_id there is just the absolute row position (`.I`) in the raw
# llama3_70b _ann.parquet log, which is laid out as one block per variable,
# each block in original narrative order. The qwen/commandrp logs use the
# newer (row, variable, attempt) schema, but their attempt == 1 pass
# processes every row for every variable in that exact same order — so
# recomputing `.I` after filtering to attempt == 1 reproduces the identical
# row_id scheme, letting us join on (row_id, variable, gen_model) directly.
dataset <- "nsduh"
hitl <- fread("results/manual_labels.csv")[!is.na(human) & human != ""]

#' Absolute row position after restricting to attempt 1 (a no-op for logs
#' without an attempt column, i.e. the old single-attempt llama3_70b format).
row_id_log <- function(model, ann_model) {
  fl  <- f("data/cache/{dataset}_{model}_{ann_model}__ann.parquet")
  log <- as.data.table(read_parquet(fl))
  if ("attempt" %in% names(log)) log <- log[attempt == 1]
  log[, .(row_id = .I, variable, response, prompt)]
}

full_log <- function(ann_model) {
  rbindlist(lapply(unique(hitl$gen_model), function(g)
    row_id_log(g, ann_model)[, gen_model := g]))
}

# --- sanity check: re-derived llama3_70b responses must match manual_labels' -
# --- own "llm" column before trusting this scheme for the new annotators ----
check <- merge(hitl[, .(row_id, variable, gen_model, llm)], full_log("llama3_70b"),
               by = c("row_id", "variable", "gen_model"))
stopifnot("row_id reconstruction doesn't match manual_labels.csv" =
           all(check$llm == check$response))
message("Row-id reconstruction verified on ", nrow(check), " / ", nrow(hitl), " rows.")

# --- attach qwen / commandrp attempt-1 responses, score vs. human ----------
hitl <- merge(hitl, check[, .(row_id, variable, gen_model, prompt)],
             by = c("row_id", "variable", "gen_model"))
hitl <- merge(hitl, full_log("qwen25_72b")[, .(row_id, variable, gen_model, qwen25_72b = response)],
             by = c("row_id", "variable", "gen_model"), all.x = TRUE)
hitl <- merge(hitl, full_log("commandrp_104b")[, .(row_id, variable, gen_model, commandrp_104b = response)],
             by = c("row_id", "variable", "gen_model"), all.x = TRUE)


# adjust the changed age answers...
hitl[variable == "age", qwen25_72b := LETTERS[letter_to_int(qwen25_72b) + 3]]
hitl[variable == "age", commandrp_104b := LETTERS[letter_to_int(commandrp_104b) + 3]]

result <- data.table(
  annotator = c("llama3_70b (original)", "qwen25_72b", "commandrp_104b"),
  n = c(sum(!is.na(hitl$llm)), sum(!is.na(hitl$qwen25_72b)), sum(!is.na(hitl$commandrp_104b))),
  accuracy = c(
    mean(hitl$llm == hitl$human, na.rm = TRUE),
    mean(hitl$qwen25_72b == hitl$human, na.rm = TRUE),
    mean(hitl$commandrp_104b == hitl$human, na.rm = TRUE)
  )
)
print(result)

# --- print the prompt + all 4 answers for every disagreement ----------------
disagree <- hitl[!(llm == qwen25_72b & qwen25_72b == commandrp_104b & commandrp_104b == human)]

for (i in seq_len(nrow(disagree))) {
  r <- disagree[i]
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("row_id:", r$row_id, "| variable:", r$variable, "| gen_model:", r$gen_model, "\n")
  cat(strrep("-", 70), "\n", sep = "")
  cat(r$prompt, "\n")
  cat(strrep("-", 70), "\n", sep = "")
  cat("Human:", r$human, "| llama3_70b:", r$llm,
      "| qwen25_72b:", r$qwen25_72b, "| commandrp_104b:", r$commandrp_104b, "\n")
  browser()
}

letter_to_int <- function(x) match(x, LETTERS)
