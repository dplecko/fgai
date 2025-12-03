
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), 
                 source))

# load the data
df <- as.data.frame(read_parquet("data/nsduh_envs_llama3_8b.parquet"))

head(df)

# subset to relevant attribute
df <- df[df$race %in% c("Black", "White"), ]
df$race <- as.integer(df$race == "Black")

# inspect features that cause non-overlap
ggplot(
  melt.data.table(as.data.table(df), id.vars = c("race", "env")),
  aes(x = value, fill = factor(env))
) + geom_histogram(aes(y = after_stat(density)), position = "dodge", alpha = 0.5) + theme_bw() +
  facet_wrap(~variable, scales = "free")

# construct the SFM
X <- "race"
Z <- c("age", "sex")
W <- c("edu", "income")
Y <- c("mj_monthly")
E <- "env"

# perform the causal decomposition
res <- one_step_debias(df, X, Z, W, Y, E, eps_trim = 0.001)

plot_one <- function(res) {
  
  res2 <- as.data.table(res)
  res2 <- res2[!grepl("delta", measure)]
  res2[, meas := gsub("-.*", "", measure)]
  res2[, env := gsub(".*-", "", measure)]
  
  pd <- position_dodge(width = 0.9)
  
  res2[, Measure := factor(meas, levels = c("tv", "ctfde", "ctfie", "ctfse"),
                        labels = c("TV", "Ctf-DE", "Ctf-IE", "Ctf-SE"))]
  
  res2[, Environment := factor(env, levels = c("world", "model"), 
                       labels = c("World", "Model"))]
  ggplot(res2,aes(x = Measure, y = value, fill = Measure, 
                  pattern = Environment, group = Environment)) +
    geom_col_pattern(
      position = pd,
      colour = "black",
      linewidth = 1,
      pattern_fill = "black",
      pattern_density = 0.2,
      pattern_spacing = 0.05
    ) +
    geom_errorbar(
      aes(ymin = value - 1.96 * sd,
          ymax = value + 1.96 * sd),
      position = pd,
      width = 0.25
    ) +
    theme_bw() + scale_y_continuous(labels = scales::percent) +
    theme(legend.position = "inside", legend.position.inside = c(0.65, 0.85),
          legend.direction = "horizontal", legend.box.background = element_rect())
}

plot_one(res)

plot_three <- function(res) {
  
  res <- as.data.table(res)
  
  res[grepl("de-", measure), group := c("DE")]
  res[grepl("ie-", measure), group := c("IE")]
  res[grepl("se-", measure), group := c("SE")]
  
  res[, env := gsub(".*-", "", measure)]
  
  res[, env := factor(env, levels = c("world", "model", "fy", "fw", "fz"),
                      labels = c("World", "Model", "f_Y", "f_W", "f_Z"))]
  
  plotlist <- list()
  for (grp in c("DE", "IE", "SE")) {
    
    cres <- res[group == grp]
    cres[3:5, value := -value]
    plotlist[[grp]] <- waterfall_plot(
      comp = c(paste0("World (", grp, ")"), "Delta f_Y", "Delta f_W", 
               "Delta f_Z", paste0("Model (", grp, ")")),
      val = c(cres$value[c(1, 3, 4, 5)], 0),
      sd = c(cres$sd[c(1, 3, 4, 5, 2)])
    )
  }
  
  plotlist
  # cowplot::plot_grid(plotlist = plotlist, labels = c("A", "B", "C"), ncol=3)
}

wfalls <- plot_three(res)

ggsave("results/tv-decomps-nsduh-llama3-8b.png", plot = plot_one(res),
       width=7, height=4)

for (grp in c("DE", "IE", "SE")) {
  
  plot <- wfalls[[grp]]
  if (grp == "DE") {
    plot <- plot + theme(legend.position = "inside", 
                         legend.position.inside = c(0.5,0.1),
                         legend.direction = "horizontal",
                         legend.box.background = element_rect()) +
      guides(
        fill   = guide_legend(nrow = 1, byrow = TRUE),
        colour = guide_legend(nrow = 1, byrow = TRUE)
      )
  } else plot <- plot + theme(legend.position = "none")
  ggsave(paste0("results/wfall-", grp, "-nsduh-llama3-8b.png"), plot = plot,
         width=7, height=4)
}

