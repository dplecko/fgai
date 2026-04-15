# --- TV decomposition bar chart ----------------------------------------------

#' @param eff output of extract_stage_effects (12 rows, sign-flipped)
plot_one <- function(eff) {
  
  world <- eff[stage == "world"]
  model <- eff[stage == "model"]
  
  # TV = sum of sign-flipped effects
  dt <- rbind(
    data.table(ce = "tv", value = sum(world$value),
               sd = sqrt(sum(world$sd^2)), env = "World"),
    world[, .(ce, value, sd, env = "World")],
    data.table(ce = "tv", value = sum(model$value),
               sd = sqrt(sum(model$sd^2)), env = "Model"),
    model[, .(ce, value, sd, env = "Model")]
  )
  
  dt[, Measure := factor(ce, levels = c("tv", "de", "ie", "se"),
                         labels = c("TV", "DE", "IE", "SE"))]
  dt[, Environment := factor(env, levels = c("World", "Model"))]
  
  pd <- position_dodge(width = 0.9)
  
  ggplot(dt, aes(x = Measure, y = value, fill = Measure,
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
    scale_x_continuous(breaks = seq_along(lev), labels = lev) +
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
          list(alpha = 0, colour = NA),
          list(), list(),
          list(alpha = 0, colour = NA),
          list(), list()
        )
      )
    ) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = "Effect Value") +
    theme_bw() +
    theme(legend.text = element_markdown())
}

# --- three waterfall plots (DE, IE, SE) --------------------------------------

#' @param eff output of extract_stage_effects (12 rows, sign-flipped)
plot_three <- function(eff) {
  
  stages_ord  <- c("world", "fy", "fw", "model")
  mech_labels <- c("Delta f_Y", "Delta f_W", "Delta f_Z")
  
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