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
  summary[, proportion := {
    raw_pct <- 100 * count / total
    pct <- floor(raw_pct)
    remainder <- if (sum(count) > 0) 100 - sum(pct) else 0
    if (remainder > 0) {
      add_to <- order(raw_pct - pct, decreasing = TRUE)[seq_len(remainder)]
      pct[add_to] <- pct[add_to] + 1
    }
    pct / 100
  }, by = .(model, direction)]
  
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
    tabular_spec  <- "l ccc ccc"
    multicol_line <- " & \\multicolumn{3}{c}{\\textbf{Disadvantage}} & \\multicolumn{3}{c}{\\textbf{Advantage}} \\\\"
    cmidrule_line <- "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}"
    header_line   <- "\\textbf{Model} & Amplify & Dampen & Reverse & Amplify & Dampen & Reverse \\\\"
  } else {
    tabular_spec  <- "l ccc"
    multicol_line <- NULL
    cmidrule_line <- NULL
    header_line   <- "\\textbf{Model} & Amp. & Damp. & Rev. \\\\"
  }
  
  lines <- c(
    # "\\begin{table}[t]",
    "\\centering",
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
  
  lines <- c(
    lines, 
    "\\bottomrule", 
    "\\end{tabular}", 
    f("\\captionof{table}{<<caption>>}", .open = "<<", .close = ">>"),
    f("\\label{<<label>>}", .open = "<<", .close = ">>")
    # "\\end{table}"
  )
  writeLines(paste(lines, collapse = "\n"), file)
  message("Written: ", file)
}

# --- real-world discrimination table ----------------------------------------

#' Summary table of DE/IE/SE in the real world, per dataset.
#' Cells: value% $\pm$ sd% (rounded 1dp).
#' Color: blue = disadvantage (value > 0), green = advantage (value < 0).
#'        lighter shade if |value| < 1.96 * sd (not significant).
#'
#' Requires \usepackage{booktabs, colortbl, xcolor}.
write_world_latex <- function(eff, file,
                              caption = "Real-world discrimination with confidence intervals.",
                              label = "tab:world") {
  
  dt <- unique(eff[stage == "world" & ce %in% c("de", "ie", "se"),
                   .(dataset, ce, value, sd)])
  
  # rename datasets
  dt[, dataset := DATASET_NAMES[dataset]]
  
  fmt_cell <- function(v, s) {
    sig   <- abs(v) >= 1.96 * s
    col   <- if (v > 0) "blue" else "green"
    shade <- if (sig) 40 else 15
    sprintf("\\cellcolor{%s!%d}$%.1f \\%%\\pm %.1f$ \\%%",
            col, shade, 100 * v, 100 * 1.96 * s)
  }
  
  wide <- dcast(dt, dataset ~ ce, value.var = c("value", "sd"))
  
  datasets_used <- unique(dt$dataset)
  
  lines <- c(
    "\\begin{table}[t]",
    "\\centering",
    "\\setstretch{1.4}",
    "\\begin{tabular}{l | c|c|c}",
    "\\toprule",
    "\\textbf{Dataset} & Direct & Indirect & Spurious \\\\",
    "\\midrule"
  )
  
  for (ds in datasets_used) {
    row <- wide[dataset == ds]
    cells <- sapply(c("de", "ie", "se"), function(c)
      fmt_cell(row[[paste0("value_", c)]], row[[paste0("sd_", c)]]))
    label_ds <- gsub("_", "\\_", ds, fixed = TRUE)
    lines <- c(lines, paste0(label_ds, " & ", paste(cells, collapse = " & "), " \\\\"))
  }
  
  lines <- c(
    lines, "\\bottomrule", 
    "\\end{tabular}",
    "\\vspace{0.1in}",
    f("\\caption{<<caption>>}", .open = "<<", .close = ">>"),
    f("\\label{<<label>>}",    .open = "<<", .close = ">>"),
    "\\end{table}"
  )
  writeLines(paste(lines, collapse = "\n"), file)
  message("Written: ", file)
}
