
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

datasets <- c("nsduh", "brfss")
models   <- c("llama3_8b", "qwen35_27b")

for (i in seq_along(datasets)) {
  
  dataset <- datasets[i]
  model <- models[i]
  
  # SFM
  sfm <- load_sfm(dataset)
  X <- sfm$X; Z <- sfm$Z; W <- sfm$W; Y <- sfm$Y
  
  # --- estimate and extract ----------------------------------------------------
  
  df_lst <- load_model_data(dataset, model)
  eff    <- estimate_within(df_lst, X, Z, W, Y,
                            dataset = dataset, model = model)
  
  # --- TV decomposition plot ---------------------------------------------------
  
  ggsave(f("results/tv-decomps-{dataset}-{model}.png"),
         plot = tv_plot(eff), width = 7, height = 4)
  
  # --- waterfall plots ---------------------------------------------------------
  
  wfalls <- plot_three(eff)
  for (grp in c("DE", "IE", "SE")) {
    
    plot <- wfalls[[grp]]
    if ((grp == "SE" & dataset == "nsduh") || grp == "DE" & dataset == "brfss") {
      plot <- plot + theme(
        legend.position = "inside",
        legend.position.inside = c(0.5, 0.06),
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
}
