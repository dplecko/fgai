
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- annotator sensitivity: Qwen2.5 72B vs. Command R+ 104B ---------------
# Set this to either option to produce only one heatmap. The default writes
# both the raw absolute shift and the shift normalized by the pooled SD.
HEATMAP_METRICS <- c("absolute", "sd_normalized")

ANN_A <- "llama3_70b"
ANN_B <- "commanda_111b"

DATASET_SHORT <- c(nsduh = "NSDUH", brfss = "BRFSS", census_income = "Census")
CE_LABEL      <- c(de = "Direct", ie = "Indirect", se = "Spurious")
STAGE_LABEL   <- c(fy = "$f_Y$", fw = "$f_W$", model = "$f_{X, Z}$")
MODELS <- setdiff(MODELS, c("gemma3_12b"))

stopifnot(
  all(HEATMAP_METRICS %in% c("absolute", "sd_normalized")),
  length(ANN_A) == 1L,
  length(ANN_B) == 1L,
  ANN_A != ANN_B
)

#' Estimate all non-world causal effects for one annotator.
#' estimate_within() handles caching, so rerunning this script only computes
#' dataset/model/annotator combinations that are not already cached.
get_eff <- function(dataset, model, ann_model) {
  message("\n", dataset, "/", model, "/", ann_model)
  sfm    <- load_sfm(dataset)
  df_lst <- load_model_data(dataset, model, ann_model = ann_model)
  estimate_within(
    df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
    dataset = dataset, model = model, ann_model = ann_model
  )[stage %in% names(STAGE_LABEL) & ce %in% names(CE_LABEL),
    .(dataset, model, ce, stage, value, sd)]
}

# --- gather every dataset x generative-model combination -------------------
all_eff <- rbindlist(lapply(DATASETS, function(dataset) {
  rbindlist(lapply(MODELS, function(model) {
    rbindlist(list(
      get_eff(dataset, model, ANN_A)[, annotator := ANN_A],
      get_eff(dataset, model, ANN_B)[, annotator := ANN_B]
    ))
  }))
}))

# --- compare annotators, one row per dataset x model x effect x stage ------
wide <- dcast(
  all_eff,
  dataset + model + ce + stage ~ annotator,
  value.var = c("value", "sd")
)

value_a <- paste0("value_", ANN_A)
value_b <- paste0("value_", ANN_B)
sd_a    <- paste0("sd_", ANN_A)
sd_b    <- paste0("sd_", ANN_B)

wide[, `:=`(
  abs_diff  = abs(get(value_b) - get(value_a)),
  pooled_sd = sqrt(get(sd_a)^2) # sqrt(get(sd_a)^2 + get(sd_b)^2)
)]
wide[, sd_normalized := fifelse(
  pooled_sd > 0,
  abs_diff / pooled_sd,
  fifelse(abs_diff == 0, 0, NA_real_)
)]
wide[, unstable :=
       abs(get(value_a)) > 1 |
       abs(get(value_b)) > 1
]

# Explicit factor levels preserve dataset facets and effect -> stage ordering on x.
x_grid <- rbindlist(lapply(DATASETS, function(dataset) {
  rbindlist(lapply(names(CE_LABEL), function(ce) {
    data.table(dataset = dataset, ce = ce, stage = names(STAGE_LABEL))
  }))
}))
x_grid[, `:=`(
  x_key = paste(dataset, ce, stage, sep = "__"),
  x_label = paste(CE_LABEL[ce], STAGE_LABEL[stage])
)]

wide[, x_key := paste(dataset, ce, stage, sep = "__")]
wide <- merge(wide, x_grid, by = c("dataset", "ce", "stage", "x_key"))
wide[, `:=`(
  dataset = factor(dataset, levels = DATASETS),
  x_label = factor(x_label, levels = unique(x_grid$x_label)),
  model   = factor(model, levels = rev(MODELS), labels = rev(MODEL_NAMES[MODELS]))
)]

# --- heatmaps ---------------------------------------------------------------
plot_annotator_heatmap <- function(metric = c("absolute", "sd_normalized")) {
  metric <- match.arg(metric)
  plot_dt <- copy(wide[unstable==FALSE])

  if (metric == "absolute") {
    plot_dt[, heat_value := abs_diff]
    fill_label <- "Absolute difference"
    value_labels <- scales::label_percent(accuracy = 0.1)(plot_dt$heat_value)
    fill_scale <- scale_fill_distiller(
      palette = "YlOrRd", direction = 1, labels = scales::label_percent(accuracy = 0.1)
    )
  } else {
    plot_dt[, heat_value := sd_normalized]
    fill_label <- "|Difference|\n(SD units)"
    value_labels <- sprintf("%.1f", plot_dt$heat_value)
    fill_scale <- scale_fill_gradientn(
      colours = c("#63BE7B", "#E8E3D5", "#F4A261", "#D73027"),
      values = scales::rescale(c(0, 1, 2, 3), from = c(0, 3)),
      limits = c(0, 3),
      breaks = 0:3,
      labels = c("0", "1 SD", "2 SD", "≥3 SD"),
      oob = scales::squish
    )
  }

  plot_dt[, cell_label := value_labels]

  ggplot(plot_dt, aes(x_label, model, fill = heat_value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = cell_label), size = 2.5) +
    facet_grid(~ dataset, labeller = labeller(dataset = as_labeller(DATASET_SHORT))) +
    scale_x_discrete(labels = TeX(levels(plot_dt$x_label))) +
    fill_scale +
    labs(
      x = "Effect / stage",
      y = "Generative model",
      fill = fill_label
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1),
      panel.grid = element_blank()
    )
}

plots <- setNames(lapply(HEATMAP_METRICS, plot_annotator_heatmap), HEATMAP_METRICS)
cowplot::plot_grid(plotlist = plots)

dir.create("results", showWarnings = FALSE, recursive = TRUE)
for (metric in names(plots)) {
  ggsave(
    f("results/annotator-sensitivity-{metric}.png"),
    plots[[metric]], width = 18, height = 7
  )
}

print(plots[[HEATMAP_METRICS[1]]])
