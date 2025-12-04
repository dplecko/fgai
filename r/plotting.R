
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

waterfall_plot <- function(comp, val, sd) {
  stopifnot(length(comp) == length(val))
  
  dt <- data.table(comp = comp, val = val, sd = sd)
  
  lev <- dt$comp
  dt[, comp := factor(comp, levels = lev)]
  
  dt[, end   := cumsum(val)]
  dt[, start := shift(end, fill = 0)]
  
  dt[, `:=`(
    ymin = pmin(start, end),
    ymax = pmax(start, end),
    role = "mid"
  )]
  
  dt[1,  role := "first"]
  dt[.N, role := "last"]
  
  dt[, fill_cat := fcase(
    role == "first", "World",
    role == "last",  "Model",
    val  >= 0,       "Positive",
    default =        "Negative"
  )]
  
  # bar heights: first & last from 0
  dt[, `:=`(
    ymin_bar = ymin,
    ymax_bar = ymax
  )]
  dt[1,  `:=`(ymin_bar = 0, ymax_bar = end)]
  dt[.N, `:=`(ymin_bar = 0, ymax_bar = end)]
  
  dt[, x_num  := as.numeric(comp)]
  dt[, x_next := shift(x_num, type = "lead")]
  dt[, y_mid  := (ymin_bar + ymax_bar) / 2]
  
  dt[, fill := fifelse(role == "first", "first",
                       fifelse(role == "last", "last",
                               fifelse(val >= 0, "pos", "neg")))]
  
  ggplot(dt) +
    geom_rect(
      aes(xmin = x_num - 0.4,
          xmax = x_num + 0.4,
          ymin = ymin_bar,
          ymax = ymax_bar,
          fill = fill_cat),
      colour = "black"
    ) +
    geom_segment(
      data = dt[!is.na(x_next)],
      aes(x = x_num + 0.4,
          xend = x_next - 0.4,
          y = end,
          yend = end),
      linewidth = 0.5
    ) +
    geom_errorbar(
      aes(x = x_num, y = end,
          ymin = end - sd, ymax = end + sd),
      width = 0.2, color = "gray"
    ) +
    geom_text(
      data = dt[role == "mid"],
      aes(x = x_num, y = y_mid, label = sprintf("%.2f", val)),
      vjust = 0.5
    ) +
    geom_text(
      aes(x = x_num, y = ymax_bar, label = sprintf("%.2f", end)),
      vjust = -0.4, hjust=1.2
    ) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    scale_x_continuous(breaks = seq_along(lev), labels = lev) +
    # scale_fill_manual(values = c(
    #   first = "#003f5c",
    #   pos   = "#4CAF50",
    #   neg   = "#F44336",
    #   last  = "#ffa600"
    # )) +
    scale_fill_manual(
      name   = NULL,
      limits = c(
        "Environment",          # header 1 (fake)
        "World",
        "Model",
        # " ",                   # spacer (fake)
        "Contribution",   # header 2 (fake)
        "Positive",
        "Negative"
      ),
      values = c(
        "Environment"        = NA,          # invisible key
        "World"         = "#003f5c",
        "Model"          = "#ffa600",
        # " "                 = NA,          # invisible spacer
        "Contribution" = NA,          # invisible key
        "Positive"          = "#2e7d32",
        "Negative"          = "#c62828"
      ),
      labels = c("**Environment**", "World", "Model", 
                 # " ", 
                 "**Contribution**",
                 "Positive", "  Negative"),
      drop = FALSE,
      guide = guide_legend(
        override.aes = list(
          list(alpha = 0, colour = NA),  # "First/Last" header
          list(),                        # First bar
          list(),                        # Last bar
          list(alpha = 0, colour = NA),  # spacer
          list(alpha = 0, colour = NA),  # "Positive/Negative" header
          list(),                        # Positive
          list()                         # Negative
        )
      )
    ) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = "Effect Value") +
    theme_bw() +
    theme(
      legend.text = element_markdown()
    )
}

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

# waterfall_plot(
#   comp = c("Main", "D1", "D2", "Mid", "D3", "D4", "D5", "End"),
#   val  = c(50, 40, 30, 0, 10, 20, -50, 0),
#   sd   = c(5, 3, 4, 2, 3, 3, 6, 5)
# )
