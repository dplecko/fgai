
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- prompt-format sensitivity: narrative (story) vs. bulleted (record) ------
# Same generator, seeds, causal graph, and estimation; only the prompt style
# given to the generator changes. Annotator is qwen25_72b for both styles.
ann_model <- "qwen25_72b"

case_studies <- list(
  list(dataset = "nsduh", model = "gemma3_27b", label = "NSDUH–Gemma 3 27B"),
  list(dataset = "brfss", model = "qwen35_27b", label = "BRFSS–Qwen 3.5 27B")
)

CE_LABEL    <- c(de = "Direct", ie = "Indirect", se = "Spurious")
STAGE_LABEL <- c(fy = "$f_Y$", fw = "$f_W$", model = "$f_{X,Z}$")

get_eff <- function(dataset, model, style) {
  sfm    <- load_sfm(dataset)
  df_lst <- load_model_data(dataset, model, ann_model = ann_model, style = style)
  estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                  dataset = dataset, model = model, ann_model = ann_model,
                  style = style)
}

#' 9-row (ce x stage) comparison of the narrative vs. record decompositions
#' for one case study, world stage excluded (it's real data, style-invariant).
build_comparison <- function(cs) {
  eff_story  <- get_eff(cs$dataset, cs$model, "story")
  eff_record <- get_eff(cs$dataset, cs$model, "record")

  vec <- function(eff) eff[stage != "world" & ce %in% names(CE_LABEL),
                           .(ce, stage, value)]

  cmp <- merge(vec(eff_story), vec(eff_record), by = c("ce", "stage"),
              suffixes = c("_narr", "_rec"))
  cmp[, abs_shift := abs(value_rec - value_narr)]

  cmp[, ce    := factor(ce, levels = names(CE_LABEL))]
  cmp[, stage := factor(stage, levels = names(STAGE_LABEL))]
  setorder(cmp, ce, stage)
  cmp
}

write_prompt_sensitivity_md <- function(cmp, label, file, append = FALSE) {
  lines <- c(
    f("### {label}"),
    "",
    paste0("| Pathway | Mechanism Replacement | Narrative Prompt (%) | ",
           "Bulleted Prompt (%) | Absolute Shift (%) |"),
    "|---|---|---:|---:|---:|"
  )
  for (i in seq_len(nrow(cmp))) {
    row <- cmp[i]
    lines <- c(lines, sprintf(
      "| %s | %s | %.1f | %.1f | %.1f |",
      CE_LABEL[[as.character(row$ce)]], STAGE_LABEL[[as.character(row$stage)]],
      100 * row$value_narr, 100 * row$value_rec, 100 * row$abs_shift
    ))
  }
  lines <- c(lines, sprintf(
    "| **Aggregate** |  |  |  | **Mean: %.1f%%** |", 100 * mean(cmp$abs_shift)
  ), "")

  con <- file(file, open = if (append) "a" else "w")
  writeLines(lines, con)
  close(con)
  message("Written: ", label)
}

dir.create("results", showWarnings = FALSE, recursive = TRUE)
out_file <- "results/prompt-sensitivity.md"

comparisons <- list()
for (i in seq_along(case_studies)) {
  cs <- case_studies[[i]]
  message(f("[{cs$label}] narrative vs. bulleted-record"))
  cmp <- build_comparison(cs)
  comparisons[[cs$label]] <- cmp
  write_prompt_sensitivity_md(cmp, cs$label, out_file, append = (i > 1))
}

print(comparisons)
