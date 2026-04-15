
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

dataset <- "nsduh"
model   <- "llama3_8b_instruct"

# SFM
sfm <- load_sfm(dataset)
X <- sfm$X; Z <- sfm$Z; W <- sfm$W; Y <- sfm$Y

# --- estimate and extract ----------------------------------------------------

df_lst <- load_model_data(dataset, model)
meas   <- estimate_within(df_lst, X, Z, W, Y,
                          dataset = dataset, model = model)
eff    <- extract_stage_effects(meas)

# --- TV decomposition plot ---------------------------------------------------

ggsave(f("results/tv-decomps-{dataset}-{model}.png"),
       plot = plot_one(eff), width = 7, height = 4)

# --- waterfall plots ---------------------------------------------------------

wfalls <- plot_three(eff)
for (grp in c("DE", "IE", "SE")) {
  
  plot <- wfalls[[grp]]
  if (grp == "DE") {
    plot <- plot + theme(
      legend.position = "inside",
      legend.position.inside = c(0.5, 0.1),
      legend.direction = "horizontal",
      legend.box.background = element_rect()
    ) +
      guides(
        fill   = guide_legend(nrow = 1, byrow = TRUE),
        colour = guide_legend(nrow = 1, byrow = TRUE)
      )
  } else {
    plot <- plot + theme(legend.position = "none")
  }
  ggsave(f("results/wfall-{grp}-{dataset}-{model}.png"),
         plot = plot, width = 7, height = 4)
}