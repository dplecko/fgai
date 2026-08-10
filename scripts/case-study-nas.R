
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- reliability check: rows with a still-unresolved (NA) variable after ---
# --- all annotation attempts, read straight from the *_ann.parquet logs ---
# (columns: row, variable, attempt, prompt, response — see
# generation.py::annotate_data). Paths are custom on purpose: fill these in
# once the relevant _ann files are placed.
ann_files <- c(
  # "NSDUH–Gemma 3 27B"  = "~/Desktop/nsduh_gemma3_27b_qwen25_72b_XZ_ann.parquet",
  # "BRFSS–Qwen 3.5 27B" = "~/Desktop/brfss_qwen35_27b_qwen25_72b__ann.parquet"
  "NSDUH–Gemma 3 27B"  = "~/Desktop/nsduh_gemma3_27b_commandrp_104b_XZ_ann.parquet",
  "BRFSS–Qwen 3.5 27B" = "~/Desktop/brfss_qwen35_27b_commandrp_104b__ann.parquet"
)

#' Share of rows with at least one variable coming back "Answer not
#' available" on attempt 1 — i.e. how the old (single-attempt) pipeline
#' would have fared. The NA letter for each record is read off its own
#' `prompt` text (prepare_answers() always appends "Answer not available"
#' as the last option), so no var_dict lookup is needed.
check_nas <- function(path) {
  log <- as.data.table(read_parquet(path))
  log <- log[attempt == 1]
  
  log[, na_letter := stringr::str_match(
    prompt,
    "([A-Z])\\.\\s*Answer not available"
  )[, 2]]
  log[, is_na := response == na_letter]
  
  print(table(log[is_na == TRUE]$variable))
  
  row_na <- log[, .(any_na = any(is_na)), by = "row"]

  data.table(
    n_rows       = nrow(row_na),
    n_affected   = sum(row_na$any_na),
    pct_affected = mean(row_na$any_na)
  )
}

for (label in names(ann_files)) {
  message(f("[{label}]"))
  print(check_nas(ann_files[[label]]))
}

as.data.table(read_parquet(ann_files[1]))
