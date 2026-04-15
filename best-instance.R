
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

datasets <- DATASETS
models   <- MODELS

# --- estimate everything -----------------------------------------------------

eff_all <- rbindlist(lapply(datasets, function(ds) {
  sfm <- load_sfm(ds)
  rbindlist(lapply(models, function(m) {
    message(f("[{ds}] Processing: {m}"))
    df_lst <- load_model_data(ds, m)
    estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                    dataset = ds, model = m)
  }))
}))

#' Find surprising {model, dataset, stage, ce} instances across three classes:
#'   1. Reversal: largest single CE that flips sign vs world
#'   2. Amplification: largest single CE amplification
#'   3. TV-hides-bias: world |TV| < 2*sd, but max |CE| in model is large
#'
#' @param scores_dt output of score_eff()
#' @param eff_all   the underlying combined effects table (incl. world rows + sd)
#' @param k         top-k per class
#' @return data.table: class, model, dataset, stage, ce, score
find_surprises <- function(scores_dt, eff_all, k = 100) {
  
  ce_dt <- scores_dt[ce %in% c("de", "ie", "se")]
  ce_dt[, direction := ifelse(world > 0, "disadvantage", "advantage")]
  ce_dt[dataset == "census_income", 
        direction := ifelse(world > 0, "advantage", "disadvantage")]
  
  # Class 1: largest reversal
  c1 <- ce_dt[classification == "reverse"][order(-abs(score))][, head(.SD, k)]
  c1[, `:=`(class = "reversal", score = abs(score))]
  
  # Class 2: largest amplification
  c2 <- ce_dt[classification == "amplify"][order(-abs(score))][, head(.SD, k)]
  c2[, `:=`(class = "amplification", score = abs(score))]
  
  # Class 3: any-stage TV statistically near zero, but model CE large
  tv_dt <- eff_all[ce == "tv", .(model, dataset, stage,
                                 tv_value = value, tv_sd = sd)]
  near_zero <- tv_dt[abs(tv_value) < 2 * tv_sd]
  
  c3_max <- ce_dt[, .SD[which.max(abs(value))],
                  by = .(model, dataset, stage)]
  c3 <- merge(c3_max, near_zero[, .(model, dataset, stage)],
              by = c("model", "dataset", "stage"))
  c3[, score := abs(value)]
  c3 <- c3[order(-score)][, head(.SD, k)]
  c3[, class := "tv_hides_bias"]
  
  cols <- c("class", "direction", "model", "dataset", "stage", "ce", "score")
  rbindlist(list(c1[, ..cols], c2[, ..cols], c3[, ..cols]))
}

# --- score and find surprises ------------------------------------------------

scores_dt <- score_eff(eff_all)
surprises <- find_surprises(scores_dt, eff_all)
print(surprises)

# Class 1
print(surprises[class == "reversal" & direction == "advantage"], topn = 161)

# Class 2
print(surprises[class == "amplification"], topn = 161)

scores_dt[model == "llama3_70b" & dataset == "nsduh" & stage == "fy"]

scores_dt[model == "qwen35_27b" & dataset == "brfss" & stage == "model"]

# Class 3
print(surprises[class == "tv_hides_bias"], topn = 161)
scores_dt[model == "gemma3_4b" & dataset == "brfss" & stage == "fw"]
