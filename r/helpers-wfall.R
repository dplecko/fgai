# --- TV decomposition bar chart ----------------------------------------------

#' @param eff output of extract_stage_effects (12 rows, sign-flipped)
tv_plot <- function(eff) {
  
  world <- eff[stage == "world"]
  model <- eff[stage == "model"]
  
  # TV = sum of sign-flipped effects
  dt <- rbind(
    world[, .(ce, value, sd, env = "World")],
    model[, .(ce, value, sd, env = "Model")]
  )
  
  dt[, Measure := factor(ce, levels = c("tv", "de", "ie", "se"),
                         labels = c("TV", "DE", "IE", "SE"))]
  dt[, Environment := factor(env, levels = c("World", "Model"))]
  
  pd <- position_dodge(width = 0.9)

  ggplot(dt, aes(x = Measure, y = value, fill = Environment,
                 group = Environment)) +
    geom_col(position = pd) +
    geom_errorbar(
      aes(ymin = value - 1.96 * sd,
          ymax = value + 1.96 * sd),
      position = pd,
      width = 0.25
    ) +
    theme_bw(base_size = 12) + scale_y_continuous(labels = scales::percent) +
    scale_fill_manual(
      values = c("World" = "#003f5c", "Model" = "#ffa600")
    ) + ylab("Value") +
    theme(legend.position = "inside", legend.position.inside = c(0.65, 0.15),
          legend.direction = "horizontal", legend.box.background = element_rect())
}

# --- generic waterfall -------------------------------------------------------

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
  
  dt[, `:=`(ymin_bar = ymin, ymax_bar = ymax)]
  dt[1,  `:=`(ymin_bar = 0, ymax_bar = end)]
  dt[.N, `:=`(ymin_bar = 0, ymax_bar = end)]
  
  dt[, x_num  := as.numeric(comp)]
  dt[, x_next := shift(x_num, type = "lead")]
  dt[, y_mid  := (ymin_bar + ymax_bar) / 2]
  
  ggplot(dt) +
    geom_rect(
      aes(xmin = x_num - 0.4, xmax = x_num + 0.4,
          ymin = ymin_bar, ymax = ymax_bar,
          fill = fill_cat),
      colour = "black"
    ) +
    geom_segment(
      data = dt[!is.na(x_next)],
      aes(x = x_num + 0.4, xend = x_next - 0.4,
          y = end, yend = end),
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
      vjust = -0.4, hjust = 1.2
    ) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    scale_x_continuous(
      breaks = seq_along(lev),
      labels = parse(text = sapply(lev, function(x)
        if (inherits(x, "expression")) as.character(x) else deparse(x)))
    ) +
    scale_fill_manual(
      name   = NULL,
      limits = c("Environment", "World", "Model",
                 "Contribution", "Positive", "Negative"),
      values = c("Environment" = NA, "World" = "#003f5c", "Model" = "#ffa600",
                 "Contribution" = NA, "Positive" = "#2e7d32", "Negative" = "#c62828"),
      labels = c("**Environment**", "World", "Model",
                 "**Contribution**", "Positive", "  Negative"),
      drop = FALSE,
      guide = guide_legend(
        override.aes = list(
          fill   = c(NA,        "#003f5c", "#ffa600", NA,          "#2e7d32", "#c62828"),
          colour = c(NA,        "black",   "black",   NA,          "black",   "black"),
          alpha  = c(0,         1,         1,         0,           1,         1)
        )
      )
    ) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = "Effect Value") +
    theme_bw(base_size = 12) +
    theme(legend.text = element_markdown(),
          axis.text.x = element_text(size = 14))
}

# --- three waterfall plots (DE, IE, SE) --------------------------------------

#' @param eff output of extract_stage_effects (12 rows, sign-flipped)
plot_three <- function(eff) {
  
  stages_ord  <- c("world", "fy", "fw", "model")
  mech_labels <- c(TeX("$\\Delta f_Y$"), TeX("$\\Delta f_W$"), TeX("$\\Delta f_{X,Z}$"))
  
  plotlist <- list()
  for (effect in c("de", "ie", "se")) {
    
    grp <- toupper(effect)
    vals <- eff[ce == effect]
    vals <- vals[match(stages_ord, stage)]
    
    # consecutive deltas
    deltas    <- diff(vals$value)
    delta_sds <- sqrt(vals$sd[-1]^2 + vals$sd[-nrow(vals)]^2)
    
    plotlist[[grp]] <- waterfall_plot(
      comp = c(paste0("World (", grp, ")"), mech_labels,
               paste0("Model (", grp, ")")),
      val  = c(vals$value[1], deltas, 0),
      sd   = c(vals$sd[1], delta_sds, vals$sd[nrow(vals)])
    )
  }
  
  plotlist
}
