root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- Llama 3 70B vs Command A 111B annotator agreement, attempt 1 only, ----
# --- swept across every generative-model x dataset pair (env "") ----------
# Compare decoded categories; raw responses now contain structured output.
ann_a <- "llama3_70b"
ann_b <- "commanda_111b"
DS_SHORT <- c(nsduh = "NSDUH", brfss = "BRFSS", census_income = "Census")

MODELS <- setdiff(MODELS, c("deepseek_7b", "qwen35_9b", "gemma3_4b"))

#' Read one annotation log (env "") and restrict to attempt 1.
read_ann1 <- function(dataset, model, ann_model) {
  fl <- f("data/cache/{dataset}_{model}_{ann_model}__ann.parquet")
  log <- as.data.table(read_parquet(fl))
  if (!("row" %in% names(log))) {
    log[, row := seq_len(.N) - 1L, by = variable]
  }
  if (!("attempt" %in% names(log))) {
    log[, attempt := 1L]
  }
  log <- unique(log[attempt == 1], by = c("row", "variable"))
  log[, .(row, variable, response = as.character(predicted))]
}

# --- x-axis: every (dataset, variable) pair across X, Z, W, Y ----------------
ds_vars <- rbindlist(lapply(DATASETS, function(ds) {
  vars <- unlist(load_sfm(ds)[c("X", "Z", "W", "Y")], use.names = FALSE)
  data.table(dataset = ds, variable = vars, label = f("{DS_SHORT[ds]}: {vars}"))
}))

# --- per (model, dataset-variable) attempt-1 agreement rate -----------------
agreement_dt <- rbindlist(lapply(MODELS, function(model) {
  rbindlist(lapply(DATASETS, function(ds) {
    la <- read_ann1(ds, model, ann_a)
    lb <- read_ann1(ds, model, ann_b)
    cmp <- merge(la, lb, by = c("row", "variable"), suffixes = c("_a", "_b"))
    cmp[,
      .(
        agreement = mean(fcoalesce(
          response_a == response_b,
          is.na(response_a) & is.na(response_b)
        ))
      ),
      by = "variable"
    ][, `:=`(model = model, dataset = ds)]
  }))
}))

agreement_dt <- merge(agreement_dt, ds_vars, by = c("dataset", "variable"))
agreement_dt[, `:=`(
  model = factor(model, levels = rev(MODELS)),
  label = factor(label, levels = ds_vars$label)
)]

# --- heatmap: generative model (y) x dataset-variable (x) -------------------
# red at 70%, white at 90%, blue at 100%; interpolate in between
p <- ggplot(agreement_dt, aes(label, model, fill = agreement)) +
  geom_tile(color = "white") +
  scale_fill_gradientn(
    colours = c("#B2182B", "white", "#2166AC"),
    values = scales::rescale(c(0.7, 0.9, 1), from = c(0.7, 1)),
    limits = c(0.7, 1),
    labels = scales::percent,
    oob = scales::squish,
    name = "Agreement"
  ) +
  geom_text(
    aes(label = sprintf("%.1f", 100 * agreement)),
    size = 5,
    lineheight = 0.8
  ) +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

p

ggsave("results/annotator-agreement.png", p, width = 12, height = 7)

# --- inspect qwen vs. command r+ disagreements for one (dataset, variable, ---
# --- generative model), printing both annotators' prompts side by side -----
inspect_disagreements <- function(dataset, var, model, n = 10, seed = 1) {
  read_one <- function(ann_model) {
    fl <- f("data/cache/{dataset}_{model}_{ann_model}__ann.parquet")
    log <- as.data.table(read_parquet(fl))
    if (!("row" %in% names(log))) {
      log[, row := seq_len(.N) - 1L, by = variable]
    }
    if (!("attempt" %in% names(log))) {
      log[, attempt := 1L]
    }
    unique(
      log[
        attempt == 1 & variable == var,
        .(row, predicted = as.character(predicted), response, prompt)
      ],
      by = "row"
    )
  }

  cmp <- merge(
    read_one(ann_a),
    read_one(ann_b),
    by = "row",
    suffixes = c("_a", "_b")
  )
  agree <- fcoalesce(
    cmp$predicted_a == cmp$predicted_b,
    is.na(cmp$predicted_a) & is.na(cmp$predicted_b)
  )
  disagree <- cmp[!agree]

  if (!nrow(disagree)) {
    message("No disagreements for ", dataset, " / ", var, " / ", model)
    return(invisible(NULL))
  }

  set.seed(seed)
  sampled <- disagree[sample(.N, min(n, .N))]

  for (i in seq_len(nrow(sampled))) {
    r <- sampled[i]
    cat("\n", strrep("=", 70), "\n", sep = "")
    cat(
      "row:",
      r$row,
      "| dataset:",
      dataset,
      "| variable:",
      var,
      "| model:",
      model,
      "\n"
    )
    cat(strrep("-", 70), "\n[", ann_a, "]\n", r$prompt_a, "\n", sep = "")
    cat(strrep("-", 70), "\n[", ann_b, "]\n", r$prompt_b, "\n", sep = "")
    cat(
      strrep("-", 70),
      "\n",
      ann_a,
      ":",
      r$response_a,
      " | ",
      ann_b,
      ":",
      r$response_b,
      "\n",
      sep = ""
    )
    browser()
  }

  invisible(sampled)
}

inspect_disagreements("census_income", "salary_group", "ministral3_8b", n = 10)
