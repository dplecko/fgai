root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

ANN_MODEL <- "llama3_70b"
STAGES <- c("XZ", "XZW")
ATTEMPTS <- 1:4
N_SAMPLES <- 8192L

#' Flag rows with at least one "Answer not available" response by attempt.
check_nas <- function(path, attempts = ATTEMPTS) {
  log <- as.data.table(read_parquet(
    path,
    col_select = c("row", "variable", "attempt", "predicted")
  ))
  stopifnot(all(log$attempt %in% attempts))
  log[, is_na := is.na(predicted)]

  # Keep only rows actually processed in each attempt.
  log[, .(fail = any(is_na)), by = .(row, attempt)]
}

# --- collect row-level failures for each dataset, model, and stage ----------
fail_rows <- rbindlist(lapply(DATASETS, function(dataset) {
  rbindlist(lapply(MODELS, function(model) {
    rbindlist(lapply(STAGES, function(stage) {
      ann_path <- f(
        "data/cache/{dataset}_{model}_{ANN_MODEL}_{stage}_ann.parquet"
      )
      x_path <- model_data_path(dataset, model, stage, ann_model = ANN_MODEL)
      x_var <- load_sfm(dataset)$X

      fail <- check_nas(ann_path)
      x <- as.data.table(read_parquet(
        x_path,
        col_select = tidyselect::all_of(x_var)
      ))
      x[, row := seq_len(.N) - 1L]
      merge(fail, x[, .(row, x_group = get(x_var))], by = "row")[,
        `:=`(dataset = dataset, model = model, stage = stage)
      ]
    }))
  }))
}))

# --- summarize overall rates and disparities across X ----------------------
fail_rates <- fail_rows[,
  .(
    value = sum(fail) / N_SAMPLES,
    conditional_value = mean(fail)
  ),
  by = .(dataset, model, attempt, stage)
]

x_rates <- fail_rows[,
  .(
    overall_rate = sum(fail) / N_SAMPLES,
    conditional_rate = mean(fail)
  ),
  by = .(dataset, model, attempt, stage, x_group)
]
fail_disparities <- x_rates[,
  .(
    value = diff(range(overall_rate)),
    conditional_value = diff(range(conditional_rate))
  ),
  by = .(dataset, model, attempt, stage)
]

# Keep row and column order stable across both heatmaps.
row_levels <- as.vector(t(outer(
  STAGES,
  ATTEMPTS,
  function(stage, attempt) f("{stage} · Attempt {attempt}")
)))
format_plot_data <- function(dt) {
  copy(dt)[, `:=`(
    row_label = factor(
      paste(stage, "· Attempt", attempt),
      levels = rev(row_levels)
    ),
    model = factor(model, levels = MODELS, labels = MODEL_NAMES[MODELS]),
    dataset = factor(
      dataset,
      levels = DATASETS,
      labels = DATASET_NAMES[DATASETS]
    ),
    cell_label = paste0(
      scales::percent(value, accuracy = 0.1),
      " (",
      scales::percent(conditional_value, accuracy = 0.1),
      ")"
    )
  )]
}

# --- draw one common heatmap layout ----------------------------------------
plot_heatmap <- function(dt, legend_title, fill_scale) {
  plot_dt <- format_plot_data(dt)

  ggplot(plot_dt, aes(model, row_label, fill = value)) +
    geom_tile(color = "white") +
    geom_text(
      aes(label = cell_label),
      size = 3.6,
      color = "green"
    ) +
    facet_wrap(~dataset, ncol = 1) +
    fill_scale +
    labs(
      x = NULL,
      y = NULL,
      fill = legend_title,
      caption = paste(
        "Main value: overall rate/disparity (failures / 8,192);",
        "parentheses: conditional value among rows processed in that attempt"
      )
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
}

fail_fill <- scale_fill_gradientn(
  colours = c("white", "white", RColorBrewer::brewer.pal(9, "YlOrRd")[2:9]),
  values = scales::rescale(
    c(0, 0.02, seq(0.02, 0.20, length.out = 9)[-1]),
    from = c(0, 0.20)
  ),
  limits = c(0, 0.20),
  breaks = c(0, 0.05, 0.10, 0.20),
  labels = scales::label_percent(accuracy = 1),
  oob = scales::squish
)
disparity_fill <- scale_fill_distiller(
  palette = "YlOrRd",
  direction = 1,
  labels = scales::label_percent(accuracy = 0.1)
)

p_fail <- plot_heatmap(fail_rates, "Fail rate", fail_fill)
p_disparity <- plot_heatmap(
  fail_disparities,
  "Fail-rate\ndisparity",
  disparity_fill
)

dir.create("results", showWarnings = FALSE, recursive = TRUE)
ggsave("results/case-study-nas-fail-rates.png", p_fail, width = 13, height = 10)
ggsave(
  "results/case-study-nas-fail-disparities.png",
  p_disparity,
  width = 13,
  height = 10
)

p_fail
p_disparity
