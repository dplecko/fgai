# --- scoring and classification ---------------------------------------------

#' Add score, direction, and classification columns to an effects table.
#' Works on the combined data.table (from rbindlist of per-model eff tables),
#' using the world-stage rows as baseline.
#'
#' Direction: disadvantage (world > 0) / advantage (world < 0) / none (|world| < eps)
#' Classification: amplify / dampen / reverse / no_bias
#' Score: (CE^mod - CE^rw) / |CE^rw|, capped to [-1, 1]
#'
#' @param eff     combined data.table with columns: model, dataset, stage, ce, value, sd
#' @param eps     threshold below which |CE^rw| is "no bias"
#' @return        same table restricted to non-world stages, with added columns
score_eff <- function(eff, eps = 1e-6) {
  
  world <- eff[stage == "world", .(model, dataset, ce, world = value)]
  out   <- merge(eff[stage != "world"], world,
                 by = c("model", "dataset", "ce"))
  
  out[, direction := fifelse(abs(world) < eps, "none",
                             fifelse(world > 0, "disadvantage", "advantage"))]
  
  out[dataset == "census_income", 
      direction := fifelse(abs(world) < eps, "none",
                           fifelse(world > 0, "advantage", "disadvantage"))]
  
  out[, score := fifelse(direction == "none",
                         pmin(pmax(value / eps, -1), 1),
                         pmin(pmax((value - world) / abs(world), -1), 1))]
  
  out[, classification := fifelse(abs(world) < eps, "no_bias",
                                  fifelse(sign(value) != sign(world) & abs(value) > eps, "reverse",
                                          fifelse(abs(value) > abs(world), "amplify", "dampen")))]
  
  out[]
}

# --- LaTeX table -------------------------------------------------------------

#' Build and write the stereotype summary LaTeX table.
#'
#' Single row per model, with proportions of amplify/dampen/reverse,
#' separately for disadvantage and advantage. Cells are color-graded
#' by overall thresholds (light: <25%, medium: 25-50%, deep: >50%).
#'
#' Requires \usepackage{booktabs, colortbl, xcolor} in the LaTeX preamble.
#'
#' @param scores_dt output of score_eff()
#' @param file      path to write .tex
write_stereotype_latex <- function(scores_dt, file,
                                   caption = "Stereotype analysis summary.",
                                   label = "tab:stereotype",
                                   include_advantage = FALSE) {
  
  # ---- color macros: swap to orange by uncommenting ------------------------
  # color_light  <- "blue!10"
  # color_medium <- "blue!30"
  color_deep   <- "blue!60"
  color_light  <- "orange!10"
  color_medium <- "orange!35"
  color_deep   <- "orange!65"
  
  # ---- aggregate -----------------------------------------------------------
  dt <- scores_dt[direction != "none"]
  summary <- dt[, .(count = .N), by = .(model, direction, classification)]
  
  dirs_used <- if (include_advantage) c("disadvantage", "advantage") else "disadvantage"
  
  all_combos <- CJ(
    model = unique(scores_dt$model),
    direction = dirs_used,
    classification = c("amplify", "dampen", "reverse")
  )
  summary <- merge(all_combos, summary, all.x = TRUE,
                   by = c("model", "direction", "classification"))
  summary[is.na(count), count := 0]
  summary[, total := sum(count), by = .(model, direction)]
  summary[total == 0, total := 1]
  summary[, proportion := count / total]
  
  wide <- dcast(summary, model ~ direction + classification,
                value.var = "proportion")
  
  cls <- c("amplify", "dampen", "reverse")
  col_order <- paste0(rep(dirs_used, each = 3), "_", cls)
  
  fmt_cell <- function(x) {
    color <- if (x < 0.25) color_light
    else if (x < 0.50) color_medium
    else color_deep
    sprintf("\\cellcolor{%s}%.0f\\%%", color, 100 * x)
  }
  
  # ---- header lines depend on include_advantage ----------------------------
  models <- sort(unique(scores_dt$model))
  
  if (include_advantage) {
    tabular_spec <- "l ccc ccc"
    multicol_line <- " & \\multicolumn{3}{c}{\\textbf{Disadvantage}} & \\multicolumn{3}{c}{\\textbf{Advantage}} \\\\"
    cmidrule_line <- "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}"
    header_line <- "\\textbf{Model} & Amplify & Dampen & Reverse & Amplify & Dampen & Reverse \\\\"
  } else {
    tabular_spec <- "l ccc"
    multicol_line <- " & \\multicolumn{3}{c}{\\textbf{Disadvantage}} \\\\"
    cmidrule_line <- "\\cmidrule(lr){2-4}"
    header_line <- "\\textbf{Model} & Amplify & Dampen & Reverse \\\\"
  }
  
  lines <- c(
    "\\begin{table}[t]",
    "\\centering",
    f("\\caption{<<caption>>}", .open = "<<", .close = ">>"),
    f("\\label{<<label>>}", .open = "<<", .close = ">>"),
    f("\\begin{tabular}{<<tabular_spec>>}", .open = "<<", .close = ">>"),
    "\\toprule",
    multicol_line,
    cmidrule_line,
    header_line,
    "\\midrule"
  )
  
  for (m in models) {
    row   <- wide[model == m]
    cells <- sapply(col_order, function(col) fmt_cell(row[[col]]))
    label_m <- if (exists("MODEL_NAMES") && m %in% names(MODEL_NAMES)) {
      MODEL_NAMES[[m]]
    } else {
      gsub("_", "\\_", m, fixed = TRUE)
    }
    lines <- c(lines, paste0(label_m, " & ", paste(cells, collapse = " & "), " \\\\"))
  }
  
  lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
  writeLines(paste(lines, collapse = "\n"), file)
  message("Written: ", file)
}
