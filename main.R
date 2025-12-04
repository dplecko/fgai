
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), 
                 source))

method <- "within"
dataset <- "nsduh"
model <- "llama3_8b_instruct"

# load the data
# df <- as.data.frame(read_parquet("data/nsduh_envs_llama3_8b.parquet"))

if (method == "cross") {
  
  df_s0 <- read_parquet(f("data/{dataset}_{model}_XZWY.parquet"))
  df_s0$env <- 0
  df_s1 <- read_parquet(f("data/{dataset}_{model}_.parquet"))
  df_s1$env <- 1
  df <- rbind(as.data.frame(df_s0), as.data.frame(df_s1))
  df <- df[df$race %in% c("Black", "White"), ]
  df$race <- as.integer(df$race == "Black")
} else {
  
  envs <- c("", "XZ", "XZW", "XZWY")
  df_lst <- lapply(
    envs, 
    function(x) {
      fl <- paste0(paste0(c(dataset, model, x), collapse = "_"), ".parquet")
      df <- as.data.frame(read_parquet(file.path("data", fl)))
      df <- df[df$race %in% c("Black", "White"), ]
      df$race <- as.integer(df$race == "Black")
      df
    }
  )
  names(df_lst) <- envs
}

# construct the SFM
X <- "race"
Z <- c("age", "sex")
W <- c("edu", "income")
Y <- c("mj_monthly")
E <- "env"

# perform the causal decomposition
if (method == "cross") {
  
  res <- one_step_debias_env(df, X, Z, W, Y, E, eps_trim = 0.001)
} else {
  
  meas <- c()
  for (i in 1:3) meas <- list(meas, meas)
  for (env in envs) {
    
    sz <- if (grepl("XZ", env)) 0 else 1
    sw <- if (grepl("W", env)) 0 else 1
    sy <- if (grepl("Y", env)) 0 else 1
    if (env == "") env <- 1
    
    meas[[1+sz]][[1+sw]][[1+sy]] <- one_step_debias(df_lst[[env]], X, Z, W, Y, 
                                                    eps_trim = 0.001)
  }
  
  res_w <- copy(meas[[1]][[1]][[1]])
  res_m <- copy(meas[[2]][[2]][[2]])
  res <- rbind(
    res_w[, measure := paste0(measure, "-world")],
    res_m[, measure := paste0(measure, "-model")]
  )
  
  take_delta <- function(meas, s, sp, ce, mech) {
    
    ctfm <- paste0("ctf", ce)
    m <- copy(meas[[1+s[1]]][[1+s[2]]][[1+s[3]]][measure == ctfm])
    mp <- copy(meas[[1+sp[1]]][[1+sp[2]]][[1+sp[3]]][measure == ctfm])
    delta_meas <- paste0(c("delta", ce, mech), collapse="-")
    data.table(
      measure = delta_meas, value = mp$value - m$value, 
      sd = sqrt(mp$sd^2 + m$sd^2)
    )
  }
  for (ce in c("de", "ie", "se")) {
    
    res <- rbind(
      res,
      take_delta(meas, c(0, 0, 0), c(0, 0, 1), ce, "fy"),
      take_delta(meas, c(0, 0, 1), c(0, 1, 1), ce, "fw"),
      take_delta(meas, c(0, 1, 1), c(1, 1, 1), ce, "fz")
    )
  }
}

plot_one(res)
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

