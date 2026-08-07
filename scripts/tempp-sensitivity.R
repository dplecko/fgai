
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- decoding-policy sensitivity: temperature / top-p vs. the (1, 1) baseline ---
ann_model    <- "qwen25_72b"
CE_LABEL     <- c(de = "Direct", ie = "Indirect", se = "Spurious")
STAGE_LABEL  <- c(fy = "f_Y", fw = "f_W", model = "f_XZ")
TEMP_COLORS  <- c(`1` = "#E31A1C", `0.7` = "#FD8D3C", `0.5` = "#FED976")  # ColorBrewer YlOrRd

case_studies <- list(
  list(dataset = "nsduh", model = "gemma3_27b", label = "NSDUH–Gemma 3 27B"),
  list(dataset = "brfss", model = "qwen35_27b", label = "BRFSS–Qwen 3.5 27B")
)

#' (T, top_p) combos with all 3 annotated envs ("", "XZ", "XZW") on disk.
discover_grid <- function(dataset, model) {
  envs  <- c("", "XZ", "XZW")
  files <- unlist(lapply(envs, function(env) {
    list.files("data", pattern = f("^{dataset}_{model}_{ann_model}_{env}_t[0-9.]+_p[0-9.]+\\.parquet$"))
  }))
  caps <- regmatches(files, regexec("_t([0-9.]+)_p([0-9.]+)\\.parquet$", files))
  grid <- rbindlist(lapply(caps, function(m) data.table(temperature = as.numeric(m[2]), top_p = as.numeric(m[3]))))
  grid[, .N, by = .(temperature, top_p)][N == length(envs), .(temperature, top_p)]
}

get_eff <- function(dataset, model, temperature = 1, top_p = 1) {
  sfm    <- load_sfm(dataset)
  df_lst <- load_model_data(dataset, model, ann_model = ann_model,
                            temperature = temperature, top_p = top_p)
  estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                  dataset = dataset, model = model, ann_model = ann_model,
                  temperature = temperature, top_p = top_p)
}

# --- gather (case study x decoding policy) effects, long format --------------
all_eff <- rbindlist(lapply(case_studies, function(cs) {
  grid <- rbind(data.table(temperature = 1, top_p = 1), discover_grid(cs$dataset, cs$model))
  rbindlist(lapply(seq_len(nrow(grid)), function(j) {
    eff <- get_eff(cs$dataset, cs$model, grid$temperature[j], grid$top_p[j])
    eff[stage %in% names(STAGE_LABEL) & ce %in% names(CE_LABEL),
        .(study = cs$label, temperature = grid$temperature[j], top_p = grid$top_p[j],
          ce, stage, value, sd)]
  }))
}))

all_eff[, `:=`(
  study  = factor(study, levels = sapply(case_studies, `[[`, "label")),
  effect = factor(CE_LABEL[ce], levels = CE_LABEL),
  stage  = factor(STAGE_LABEL[stage], levels = STAGE_LABEL)
)]

# --- plot: stage (x) x effect (facet col) x study (facet row) ----------------
p <- ggplot(all_eff[abs(value) < 1], aes(stage, value, color = factor(temperature), linetype = factor(top_p),
                         group = interaction(temperature, top_p))) +
  geom_ribbon(aes(ymin = value - 1.96 * sd, ymax = value + 1.96 * sd, fill = factor(temperature)),
              alpha = 0.1, color = NA) +
  geom_line() +
  geom_point() +
  ggh4x::facet_grid2(
    study ~ effect,
    scales = "free_y",
    independent = "y"
  ) +
  labs(x = "Stage", y = "Effect value", color = "Temperature", linetype = "Top-p", fill = "Temperature") +
  scale_color_manual(values = TEMP_COLORS) +
  scale_fill_manual(values = TEMP_COLORS) +
  theme_bw() +
  scale_y_continuous(labels = scales::percent) +
  geom_hline(yintercept = 0, color = "grey", linetype = "dashed") +
  scale_x_discrete(labels = c(TeX("$f_Y$"), TeX("$f_W$"), TeX("$f_{X, Z}$"))) +
  coord_cartesian(xlim = c(1.5, 2.5)) +
  theme(axis.text.x = element_text(size = 13),
        legend.position = "bottom")

ggsave("results/tempp-sensitivity.png", p, width = 10.5, height = 5.5)

# --- summary table: mean absolute shift + sign agreement vs. baseline --------
base_dt <- all_eff[temperature == 1 & top_p == 1, .(study, ce, stage, base_value = value)]
cmp     <- merge(all_eff, base_dt, by = c("study", "ce", "stage"))

summary_dt <- cmp[, .(
  `Mean Abs. Shift`   = paste0(round(100 * mean(abs(value - base_value)), 1), "%"),
  `Sign Agreement`    = paste(sum(sign(value) == sign(base_value)), "/ 9")
), by = .(Study = study, Temperature = temperature, `Top-p` = top_p)][order(Study, -Temperature, -`Top-p`)]

writeLines(
  knitr::kable(summary_dt, format = "latex", booktabs = TRUE, digits = 3),
  "results/tempp-sensitivity.tex"
)
