
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- decoding-policy sensitivity: temperature / top-p vs. the (1.0, 1.0) baseline ---
ann_model <- "qwen25_72b"

case_studies <- list(
  list(dataset = "nsduh", model = "gemma3_27b", label = "NSDUH–Gemma 3 27B"),
  list(dataset = "brfss", model = "qwen35_27b", label = "BRFSS–Qwen 3.5 27B")
)

CE_LABEL <- c(de = "Direct", ie = "Indirect", se = "Spurious")

#' Which (T, top_p) combos have all 3 annotated envs ("", "XZ", "XZW") on disk
#' for this (dataset, model, ann_model). Mirrors the filename convention built
#' by model_data_path()/gen_suffix() in r/helpers.R, just run in reverse.
discover_decoding_grid <- function(dataset, model, ann_model) {
  envs <- c("", "XZ", "XZW")
  grid_parts <- list()
  for (env in envs) {
    prefix <- f("{dataset}_{model}_{ann_model}_{env}_t")
    files  <- list.files("data", pattern = paste0("^", prefix, "[0-9.]+_p[0-9.]+\\.parquet$"))
    if (!length(files)) next
    caps <- regmatches(files, regexec("_t([0-9.]+)_p([0-9.]+)\\.parquet$", files))
    grid_parts[[length(grid_parts) + 1]] <- rbindlist(lapply(caps, function(m) {
      data.table(temperature = as.numeric(m[2]), top_p = as.numeric(m[3]))
    }))
  }
  grid <- rbindlist(grid_parts)
  if (!nrow(grid)) return(data.table(temperature = numeric(0), top_p = numeric(0)))

  # keep only combos present under all 3 envs
  counts <- grid[, .N, by = .(temperature, top_p)]
  counts[N == length(envs), .(temperature, top_p)][order(-temperature, -top_p)]
}

#' estimate_within(), plus the overlap-trim percentage scraped from its
#' "N% of extreme P(x | z) ..." message (r/one-step-debias.R). Only fires on
#' a fresh (non-cached) computation; if the underlying results/cache/*.rds
#' already exists, no message is emitted and overlap_pct reads 0 for that
#' call — delete the relevant cache file to force a recheck.
get_eff <- function(dataset, model, temperature, top_p) {
  sfm    <- load_sfm(dataset)
  df_lst <- load_model_data(dataset, model, ann_model = ann_model,
                            temperature = temperature, top_p = top_p)

  overlap_pct <- 0
  eff <- withCallingHandlers(
    estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                    dataset = dataset, model = model, ann_model = ann_model,
                    temperature = temperature, top_p = top_p),
    message = function(m) {
      caps <- regmatches(conditionMessage(m),
                         regexec("^([0-9.]+)% of extreme P\\(x", conditionMessage(m)))[[1]]
      if (length(caps) == 2) overlap_pct <<- max(overlap_pct, as.numeric(caps[2]))
    }
  )
  attr(eff, "overlap_pct") <- overlap_pct
  eff
}

#' One row: mean absolute shift + sign agreement over the 9 model-dependent
#' stage effects (ce in {de,ie,se} x stage in {fy,fw,model}) vs. baseline.
#' overlap_pct is the worse of the two runs' trimmed-propensity share, for
#' flagging rows whose estimate rests on a thin-overlap population.
compare_to_baseline <- function(eff_base, eff_alt, temperature, top_p) {
  vec <- function(eff) eff[stage != "world" & ce %in% names(CE_LABEL),
                           .(ce, stage, value)]
  cmp <- merge(vec(eff_base), vec(eff_alt), by = c("ce", "stage"),
              suffixes = c("_base", "_alt"))

  data.table(
    temperature    = temperature,
    top_p          = top_p,
    mean_abs_shift = mean(abs(cmp$value_alt - cmp$value_base)),
    sign_agree_n   = sum(sign(cmp$value_alt) == sign(cmp$value_base)),
    n_dims         = nrow(cmp),
    overlap_pct    = max(attr(eff_base, "overlap_pct"), attr(eff_alt, "overlap_pct"))
  )
}

write_tempp_md <- function(dt, label, file, append = FALSE) {
  lines <- c(
    f("### {label}"),
    "",
    "| Temperature | Top-$p$ | Mean Absolute Shift (%) | Sign Agreement |",
    "|---:|---:|---:|---:|"
  )
  flagged <- list()
  for (i in seq_len(nrow(dt))) {
    row     <- dt[i]
    overlap <- row$overlap_pct > 0
    temp_cell <- if (overlap) {
      flagged[[length(flagged) + 1]] <- row
      sprintf('<span style="color:red">%.1f*</span>', row$temperature)
    } else {
      sprintf("%.1f", row$temperature)
    }
    lines <- c(lines, sprintf(
      "| %s | %.1f | %.1f | %d/%d |",
      temp_cell, row$top_p, 100 * row$mean_abs_shift, row$sign_agree_n, row$n_dims
    ))
  }
  if (length(flagged)) {
    lines <- c(lines, "",
              paste0("*\\* overlap issue: >2% of extreme $P(x \\mid z)$ or ",
                     "$P(x \\mid z, w)$ propensities trimmed at ",
                     "$\\epsilon_{\\text{trim}} = 0.001$; reported results are ",
                     "for the overlap population only.*"))
    for (fr in flagged) {
      lines <- c(lines, f("  - T={fr$temperature}, top-p={fr$top_p}: {round(fr$overlap_pct, 1)}% trimmed"))
    }
  }
  lines <- c(lines, "")

  con <- file(file, open = if (append) "a" else "w")
  writeLines(lines, con)
  close(con)
  message("Written: ", label)
}

dir.create("results", showWarnings = FALSE, recursive = TRUE)
out_file <- "results/tempp-sensitivity.md"

results_by_cs <- list()
for (i in seq_along(case_studies)) {
  cs <- case_studies[[i]]
  message(f("[{cs$label}] decoding-policy sensitivity"))

  eff_base <- get_eff(cs$dataset, cs$model, temperature = 1.0, top_p = 1.0)
  grid     <- discover_decoding_grid(cs$dataset, cs$model, ann_model)

  if (!nrow(grid)) {
    warning(f("No complete (T, top_p) combos found on disk for {cs$label}; skipping."))
    next
  }

  rows <- list(compare_to_baseline(eff_base, eff_base, 1.0, 1.0))
  for (j in seq_len(nrow(grid))) {
    g <- grid[j]
    message(f("  T={g$temperature}, top_p={g$top_p}"))
    eff_alt <- get_eff(cs$dataset, cs$model, g$temperature, g$top_p)
    rows[[length(rows) + 1]] <- compare_to_baseline(eff_base, eff_alt, g$temperature, g$top_p)
  }

  dt <- rbindlist(rows)
  results_by_cs[[cs$label]] <- dt
  write_tempp_md(dt, cs$label, out_file, append = (i > 1))
}

print(results_by_cs)
