
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- Qwen2.5 72B vs Command R+ 104B annotator agreement, attempt 1 only, ---
# --- swept across every generative-model x dataset pair (env "") ----------
# Comparing raw response letters (not decoded categories) is enough: for a
# given (row, variable), both annotators see the same option ordering in
# their prompt, so equal letters <=> equal chosen category.
ann_a    <- "qwen25_72b"
ann_b    <- "commandrp_104b"
DS_SHORT <- c(nsduh = "NSDUH", brfss = "BRFSS", census_income = "Census")

#' Read one annotation log (env "") and restrict to attempt 1, flagging
#' whether each response is that record's own "Answer not available" letter.
read_ann1 <- function(dataset, model, ann_model) {
  fl  <- f("data/cache/{dataset}_{model}_{ann_model}__ann.parquet")
  log <- as.data.table(read_parquet(fl))
  if (!("row" %in% names(log)))     log[, row := seq_len(.N) - 1L, by = variable]
  if (!("attempt" %in% names(log))) log[, attempt := 1L]
  log <- unique(log[attempt == 1], by = c("row", "variable"))
  log[, na_letter := stringr::str_match(prompt, "([A-Z])\\.\\s*Answer not available")[, 2]]
  log[, .(row, variable, response, is_na = response == na_letter)]
}

# --- x-axis: every (dataset, variable) pair across X, Z, W, Y ----------------
ds_vars <- rbindlist(lapply(DATASETS, function(ds) {
  vars <- unlist(load_sfm(ds)[c("X", "Z", "W", "Y")], use.names = FALSE)
  data.table(dataset = ds, variable = vars, label = f("{DS_SHORT[ds]}: {vars}"))
}))

# --- per (model, dataset-variable) attempt-1 agreement rate, split by -------
# --- whether qwen (ann_a) chose "Answer not available" or not --------------
agreement_dt <- rbindlist(lapply(MODELS, function(model) {
  rbindlist(lapply(DATASETS, function(ds) {
    la <- read_ann1(ds, model, ann_a)
    lb <- read_ann1(ds, model, ann_b)
    cmp <- merge(la, lb, by = c("row", "variable"), suffixes = c("_a", "_b"))
    cmp[, .(
      agreement        = mean(response_a == response_b),
      agreement_not_na = mean(response_a[!is_na_a] == response_b[!is_na_a]),
      agreement_na      = mean(response_a[is_na_a] == response_b[is_na_a])
    ), by = "variable"][, `:=`(model = model, dataset = ds)]
  }))
}))

agreement_dt <- merge(agreement_dt, ds_vars, by = c("dataset", "variable"))
agreement_dt[, `:=`(
  model = factor(model, levels = rev(MODELS)),
  label = factor(label, levels = ds_vars$label)
)]

# --- heatmap: generative model (y) x dataset-variable (x) -------------------
# fill = agreement when qwen did NOT say "Answer not available"; text shows
# that number on top and the NA-conditional agreement below it.
# white above 99% agreement, red at/below 95%, gradient in between
p <- ggplot(agreement_dt, aes(label, model, fill = agreement_not_na)) +
  geom_tile(color = "white") +
  scale_fill_gradientn(
    colours = c("#B2182B", "#B2182B", "white", "white"),
    values  = c(0, 0.9, 0.99, 1),
    limits  = c(0, 1),
    labels  = scales::percent,
    name    = "Agreement\n(Qwen ≠ NA)"
  ) +
  geom_text(aes(label = sprintf("%.1f\n%.1f", 100 * agreement_not_na, 100 * agreement_na)),
           size = 5, lineheight = 0.8) +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

p

ggsave("results/annotator-agreement.png", p, width = 12, height = 7)

# --- inspect qwen vs. command r+ disagreements for one (dataset, variable, ---
# --- generative model), printing both annotators' prompts side by side -----
inspect_disagreements <- function(dataset, var, model, n = 10, seed = 1) {
  read_one <- function(ann_model) {
    fl  <- f("data/cache/{dataset}_{model}_{ann_model}__ann.parquet")
    log <- as.data.table(read_parquet(fl))
    if (!("row" %in% names(log)))     log[, row := seq_len(.N) - 1L, by = variable]
    if (!("attempt" %in% names(log))) log[, attempt := 1L]
    unique(log[attempt == 1 & variable == var, 
               .(row, response, prompt)], by = "row")
  }

  cmp <- merge(read_one(ann_a), read_one(ann_b), by = "row", suffixes = c("_a", "_b"))
  disagree <- cmp[response_a != response_b]

  if (!nrow(disagree)) {
    message("No disagreements for ", dataset, " / ", var, " / ", model)
    return(invisible(NULL))
  }

  set.seed(seed)
  sampled <- disagree[sample(.N, min(n, .N))]

  for (i in seq_len(nrow(sampled))) {
    r <- sampled[i]
    cat("\n", strrep("=", 70), "\n", sep = "")
    cat("row:", r$row, "| dataset:", dataset, "| variable:", var, "| model:", model, "\n")
    cat(strrep("-", 70), "\n[", ann_a, "]\n", r$prompt_a, "\n", sep = "")
    cat(strrep("-", 70), "\n[", ann_b, "]\n", r$prompt_b, "\n", sep = "")
    cat(strrep("-", 70), "\n", ann_a, ":", r$response_a, " | ", ann_b, ":", r$response_b, "\n", sep = "")
    browser()
  }

  invisible(sampled)
}

inspect_disagreements("brfss", "age_group", "llama3_70b", n = 10)
