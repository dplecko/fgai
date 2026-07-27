
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- setup: three annotators, cross-compared on the two focused case studies ---
ANNOTATOR_SHORT <- c(
  llama3_70b     = "Llama",
  qwen25_72b     = "Qwen2.5",
  commandrp_104b = "Command R+"
)

case_studies <- list(
  list(dataset = "nsduh", model = "gemma3_27b", label = "NSDUH–Gemma 3 27B"),
  list(dataset = "brfss", model = "qwen35_27b", label = "BRFSS–Qwen 3.5 27B")
)

ann_pairs <- list(
  c("qwen25_72b", "commandrp_104b"),
  c("llama3_70b", "qwen25_72b"),
  c("llama3_70b", "commandrp_104b")
)

# --- effect decompositions per annotator (cached across the two case studies) ---
eff_cache <- new.env()

get_eff <- function(dataset, model, ann_model) {
  key <- f("{dataset}_{model}_{ann_model}")
  if (!exists(key, envir = eff_cache)) {
    sfm    <- load_sfm(dataset)
    df_lst <- load_model_data(dataset, model, ann_model = ann_model)
    eff <- estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                           dataset = dataset, model = model, ann_model = ann_model)
    assign(key, eff, envir = eff_cache)
  }
  get(key, envir = eff_cache)
}

# 9D vector per (dataset, model): ce x stage, world stage excluded
# (world uses XZWY, which is annotator-independent).
vec_dt <- function(eff) {
  eff[stage != "world" & ce %in% c("de", "ie", "se"),
      .(dataset, model, ce, stage, value)]
}

#' Mean absolute effect shift and sign agreement between two annotators,
#' across the 9 ce x stage dims of the causal decomposition.
effect_shift <- function(dataset, model, ann_a, ann_b) {
  va <- vec_dt(get_eff(dataset, model, ann_a))
  vb <- vec_dt(get_eff(dataset, model, ann_b))
  cmp <- merge(va, vb, by = c("dataset", "model", "ce", "stage"),
              suffixes = c("_a", "_b"))
  list(
    mean_abs_shift = mean(abs(cmp$value_b - cmp$value_a)),
    sign_agree     = mean(sign(cmp$value_a) == sign(cmp$value_b))
  )
}

# --- label agreement: raw annotated columns, compared row-by-row -------------
#
# For env e, the variables actually extracted by the annotator from the
# generated narrative are those NOT already fixed at their real-world value,
# i.e. setdiff({X, Z, W, Y}, letters(e)):
#   ""    -> X, Z, W, Y all annotated
#   "XZ"  -> W, Y annotated
#   "XZW" -> Y annotated
# "XZWY" is ground truth (no annotation involved) and is excluded.
#
# Row order is assumed aligned across annotators: each annotator re-labels
# the same fixed, cached set of generated narratives for a given
# (dataset, model, env), so row i is the same narrative for both.
annotated_vars <- function(sfm, env) {
  blocks  <- list(X = sfm$X, Z = sfm$Z, W = sfm$W, Y = sfm$Y)
  present <- strsplit(env, "")[[1]]
  unlist(blocks[setdiff(names(blocks), present)], use.names = FALSE)
}

read_raw <- function(dataset, model, ann_model, env) {
  fl <- paste0(paste0(c(dataset, model, ann_model, env), collapse = "_"), ".parquet")
  as.data.frame(read_parquet(file.path("data", fl)))
}

#' Average and minimum per-variable label agreement between two annotators,
#' pooled over the three annotated environments ("", "XZ", "XZW").
label_agreement <- function(dataset, model, ann_a, ann_b) {
  sfm  <- load_sfm(dataset)
  envs <- c("", "XZ", "XZW")

  rates <- unlist(lapply(envs, function(env) {
    vars <- annotated_vars(sfm, env)
    da   <- read_raw(dataset, model, ann_a, env)
    db   <- read_raw(dataset, model, ann_b, env)
    n    <- min(nrow(da), nrow(db))

    sapply(vars, function(v) {
      if (!(v %in% names(da)) || !(v %in% names(db))) return(NA_real_)
      mean(as.character(da[[v]][seq_len(n)]) == as.character(db[[v]][seq_len(n)]),
           na.rm = TRUE)
    })
  }))
  rates <- rates[!is.na(rates)]

  list(avg = mean(rates), min = min(rates))
}

# --- build the cross-comparison table -----------------------------------------
rows <- list()
for (cs in case_studies) {
  for (pair in ann_pairs) {
    a <- pair[1]; b <- pair[2]
    message(f("[{cs$label}] {ANNOTATOR_SHORT[a]} vs {ANNOTATOR_SHORT[b]}"))

    la <- label_agreement(cs$dataset, cs$model, a, b)
    es <- effect_shift(cs$dataset, cs$model, a, b)

    rows[[length(rows) + 1]] <- data.table(
      ds_model             = cs$label,
      pair_label           = f("{ANNOTATOR_SHORT[a]}–{ANNOTATOR_SHORT[b]}"),
      avg_label_agreement  = la$avg,
      min_var_agreement    = la$min,
      mean_abs_shift       = es$mean_abs_shift,
      sign_agreement       = es$sign_agree
    )
  }
}
result_dt <- rbindlist(rows)

# --- markdown table -------------------------------------------------------------
write_cross_comparison_md <- function(dt, file) {
  lines <- c(
    paste0("| Dataset–Model Pair | Annotator Pair | Average Label Agreement | ",
           "Minimum Variable Agreement | Mean Absolute Effect Shift (%) | Sign Agreement |"),
    "|---|---|---:|---:|---:|---:|"
  )
  for (i in seq_len(nrow(dt))) {
    row <- dt[i]
    lines <- c(lines, sprintf(
      "| %s | %s | %.1f%% | %.1f%% | %.1f%% | %.0f%% |",
      row$ds_model, row$pair_label,
      100 * row$avg_label_agreement, 100 * row$min_var_agreement,
      100 * row$mean_abs_shift, 100 * row$sign_agreement
    ))
  }
  writeLines(lines, file)
  message("Written: ", file)
}

dir.create("results", showWarnings = FALSE, recursive = TRUE)
write_cross_comparison_md(result_dt, "results/multi-annotator.md")

print(result_dt)
