
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- prompt-format sensitivity: narrative (story) vs. bulleted (record) ------
# Same generator, seeds, causal graph, and estimation; only the prompt style
# given to the generator changes. Annotator is qwen25_72b for both styles.
ann_model   <- "qwen25_72b"
CE_LABEL    <- c(de = "Direct", ie = "Indirect", se = "Spurious")
STAGE_LABEL <- c(fy = "f_Y", fw = "f_W", model = "f_XZ")
STYLE_LABEL <- c(story = "Narrative", record = "Bulleted")

case_studies <- list(
  list(dataset = "nsduh", model = "gemma3_27b", label = "NSDUH–Gemma 3 27B"),
  list(dataset = "brfss", model = "qwen35_27b", label = "BRFSS–Qwen 3.5 27B")
)

get_eff <- function(dataset, model, style) {
  sfm    <- load_sfm(dataset)
  df_lst <- load_model_data(dataset, model, ann_model = ann_model, style = style)
  estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                  dataset = dataset, model = model, ann_model = ann_model, style = style)
}

# --- gather (case study x style) effects, long format -------------------------
all_eff <- rbindlist(lapply(case_studies, function(cs) {
  rbindlist(lapply(names(STYLE_LABEL), function(sty) {
    eff <- get_eff(cs$dataset, cs$model, sty)
    eff[stage %in% names(STAGE_LABEL) & ce %in% names(CE_LABEL),
        .(study = cs$label, style = sty, ce, stage, value, sd)]
  }))
}))

all_eff[, `:=`(
  study  = factor(study, levels = sapply(case_studies, `[[`, "label")),
  effect = factor(CE_LABEL[ce], levels = CE_LABEL),
  stage  = factor(STAGE_LABEL[stage], levels = STAGE_LABEL)
)]

# --- narrative vs. record, one row per (study, effect, stage) ----------------
wide <- dcast(all_eff, study + effect + stage ~ style, value.var = c("value", "sd"))
wide[, `:=`(
  diff    = value_record - value_story,
  se_diff = sqrt(sd_story^2 + sd_record^2)
)]
wide[, shift_sd := diff / se_diff]  # shift relative to combined estimation uncertainty

# --- scatter: narrative vs. record, per-study correlation ---------------------
cor_dt <- wide[, .(label = sprintf("r = %.2f", cor(value_story, value_record))), by = study]

p_scatter <- ggplot(wide, aes(value_story, value_record, color = effect)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  geom_point(size = 2.5) +
  geom_text(data = cor_dt, aes(x = -Inf, y = Inf, label = label),
           inherit.aes = FALSE, hjust = -0.2, vjust = 1.5) +
  facet_wrap(~ study) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = "Narrative prompt", y = "Bulleted prompt", color = "Effect") +
  theme_bw()

ggsave("results/prompt-sens-scatter.png", p_scatter, width = 9, height = 4.5)

# --- heatmap: |shift| in SD units, stage (x) x effect (y), facet by study ----
# (SD units = shift relative to the pooled estimation SE, so a hot cell means
# the narrative/record gap is large relative to how uncertain each estimate is)
p_heat <- ggplot(wide, aes(stage, effect, fill = abs(shift_sd))) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.1f", abs(shift_sd)))) +
  facet_wrap(~ study) +
  scale_fill_distiller(palette = "YlOrRd", direction = 1) +
  labs(x = "Stage", y = "Effect", fill = "|Shift| (SD units)") +
  theme_bw() +
  scale_x_discrete(labels = c(TeX("$f_Y$"), TeX("$f_W$"), TeX("$f_{X, Z}$"))) +
  theme(axis.text.x = element_text(size = 13))

ggsave("results/prompt-sens-heat.png", p_heat, width = 9, height = 4)

# --- summary table: mean absolute shift + sign agreement, as in tempp --------
summary_dt <- wide[, .(
  `Mean Abs. Shift` = paste0(round(100 * mean(abs(diff)), 1), "%"),
  `Sign Agreement`  = paste(sum(sign(value_story) == sign(value_record)), "/ 9")
), by = .(Study = study)]

writeLines(
  knitr::kable(summary_dt, format = "latex", booktabs = TRUE),
  "results/prompt-sensitivity.tex"
)
