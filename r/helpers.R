# =============================================================================
# fgai-helpers.R — shared core for all FGAI analyses
#
# Pipeline:  load_model_data → estimate_within → extract_stage_effects
#
# Everything downstream (9D vectors, waterfall, scoring, similarity)
# reads from the 12-row stage effects table.
# =============================================================================

# --- data loading -----------------------------------------------------------

#' Format a float the way Python's f"{x}" does for the one-decimal values
#' used by --temperature/--top_p (e.g. 1 -> "1.0", 0.7 -> "0.7"). Not a
#' general float formatter; only tuned to that value set.
py_float <- function(x) {
  if (x == round(x)) sprintf("%.1f", x) else as.character(x)
}

#' Filename suffix for non-default generation settings.
#' Mirrors py/elicit.py's `gen_suffix` exactly:
#'   gen_suffix = ("" if style == "story" else f"_{style}") +
#'                ("" if temperature == 1.0 and top_p == 1.0 else f"_t{temperature}_p{top_p}")
#' Empty for the defaults (story, T=1, p=1), so paths/cache keys for existing
#' runs are unchanged.
gen_suffix <- function(style = "story", temperature = 1.0, top_p = 1.0) {

  s1 <- if (style == "story") "" else paste0("_", style)
  s2 <- if (temperature == 1.0 && top_p == 1.0) "" else {
    sprintf("_t%s_p%s", py_float(temperature), py_float(top_p))
  }
  paste0(s1, s2)
}

#' Build the data/ parquet path for one (dataset, model, env) file.
#' XZWY is ground truth: no ann_model, no style/decoding suffix (see elicit.py).
model_data_path <- function(dataset, model, env, ann_model = "llama3_70b",
                            style = "story", temperature = 1.0, top_p = 1.0) {

  if (env == "XZWY") {
    fl <- paste0(paste0(c(dataset, model, env), collapse = "_"), ".parquet")
  } else {
    suffix <- gen_suffix(style, temperature, top_p)
    fl <- paste0(paste0(c(dataset, model, ann_model, env), collapse = "_"), suffix, ".parquet")
  }
  file.path("data", fl)
}

#' Load the 4 environment datasets for a given model
#' Returns a named list keyed by env suffixes: "", "XZ", "XZW", "XZWY"
load_model_data <- function(dataset, model, ann_model = "llama3_70b", minority = TRUE,
                            style = "story", temperature = 1.0, top_p = 1.0) {

  X_var <- load_sfm(dataset)$X
  if (is.element(dataset, c("brfss", "nsduh"))) {

    if (minority) {

      X_keep <- c("Black", "Hispanic", "White")
      X_ref <- c("Black", "Hispanic")
    } else {

      X_keep <- c("Black", "White")
      X_ref <- c("Black")
    }
  } else if (grepl("census", dataset)) {

    X_keep <- c("female", "male")
    X_ref <- "female"
  }

  # "" is full generative model; XZWY is full reality;
  envs <- c("", "XZ", "XZW", "XZWY")
  df_lst <- lapply(
    envs,
    function(x) {
      fl <- model_data_path(dataset, model, x, ann_model = ann_model,
                            style = style, temperature = temperature, top_p = top_p)
      df <- as.data.frame(read_parquet(fl))
      df <- df[df[[X_var]] %in% X_keep, ]
      
      df[[X_var]] <- as.integer(df[[X_var]] %in% X_ref)
      
      if (dataset == "census_income") {
        
        df[["salary_group"]] <- factor(
          ifelse(df[["salary_group"]] >= "50001–75000 $", "no", "yes"),
          levels = c("no", "yes")
        )
      }
      
      df
    }
  )
  names(df_lst) <- envs
  df_lst
}

# --- preprocessing (internal) -----------------------------------------------

#' Convert factor columns to numeric for estimation:
#'   - binary (2 levels): yes/no → 1/0, otherwise → 0/1 codes
#'   - ordered (>2 levels): integer codes 1..K
#'   - unordered (>2 levels, not in X): one-hot, SFM updated
#'
#' @return list(data, sfm)
prepare_for_osd <- function(data, sfm) {
  
  X <- sfm$X
  
  for (col in names(data)) {
    
    if (!is.factor(data[[col]])) next
    
    cats <- levels(data[[col]])
    n_cat <- length(cats)
    
    if (n_cat == 2) {
      if (setequal(cats, c("yes", "no"))) {
        data[[col]] <- as.integer(data[[col]] == "yes")
      } else if (setequal(cats, c("Yes", "No"))) {
        data[[col]] <- as.integer(data[[col]] == "Yes")
      } else {
        data[[col]] <- as.integer(data[[col]]) - 1L
      }
      
    } else if (is.ordered(data[[col]])) {
      data[[col]] <- as.integer(data[[col]])
      
    } else if (!(col %in% X)) {
      dummies <- model.matrix(~ . - 1, data = data[, col, drop = FALSE])[, -1, drop = FALSE]
      colnames(dummies) <- gsub(" ", "_", colnames(dummies))
      
      for (part in c("X", "Z", "W", "Y")) {
        if (col %in% sfm[[part]]) {
          sfm[[part]] <- c(setdiff(sfm[[part]], col), colnames(dummies))
        }
      }
      
      data[[col]] <- NULL
      data <- cbind(data, dummies)
    }
  }
  # browser()
  list(data = data, sfm = sfm)
}

# --- estimation -------------------------------------------------------------

#' Stages of mechanism replacement, in cumulative order
STAGES <- list(
  world = list(s = c(0,0,0), env = "XZWY"),
  fy    = list(s = c(0,0,1), env = "XZW"),
  fw    = list(s = c(0,1,1), env = "XZ"),
  model = list(s = c(1,1,1), env = "")
)

#' Run one_step_debias on the 4 environment datasets and extract
#' CE values (DE, IE, SE) at each stage. IE and SE are sign-flipped
#' so all effects share DE's directionality (positive = disadvantage for X=x1).
#'
#' Results cached to results/cache/{dataset}_{model}_{ann_model}{suffix}.rds,
#' where suffix encodes non-default style/temperature/top_p (see gen_suffix());
#' empty for the defaults, so existing cache files are unaffected.
#'
#' @return data.table with 12 rows: stage × ce, columns: stage, ce, value, sd
estimate_within <- function(df_lst, X, Z, W, Y,
                            dataset = NULL, model = NULL,
                            ann_model = "llama3_70b",
                            style = "story", temperature = 1.0, top_p = 1.0,
                            eps_trim = 0.001, cache = TRUE) {

  # caching
  if (cache && !is.null(dataset) && !is.null(model)) {
    cache_dir <- "results/cache"
    suffix <- gen_suffix(style, temperature, top_p)
    cache_file <- file.path(cache_dir, f("{dataset}_{model}_{ann_model}{suffix}.rds"))
    if (file.exists(cache_file)) {
      message("Loading cached: ", model)
      return(readRDS(cache_file))
    }
  }

  sfm <- list(X = X, Z = Z, W = W, Y = Y)
  ces <- c("tv", "de", "ie", "se")
  rows <- list()

  for (stg_name in names(STAGES)) {

    env_key <- STAGES[[stg_name]]$env
    if (env_key == "") env_key <- 1
    prep <- prepare_for_osd(df_lst[[env_key]], sfm)

    with_seed(2026,
              dt <- one_step_debias(
                prep$data, prep$sfm$X, prep$sfm$Z, prep$sfm$W, prep$sfm$Y,
                eps_trim = eps_trim
              )
    )

    for (ce in ces) {

      osd_meas <- if (ce == "tv") "tv" else paste0("ctf", ce)
      row  <- dt[measure == osd_meas]
      flip <- if (ce %in% c("ie", "se")) -1 else 1
      rows[[length(rows) + 1]] <- data.table(
        stage = stg_name, ce = ce,
        value = flip * row$value, sd = row$sd
      )
    }
  }

  eff <- rbindlist(rows)
  eff[, `:=`(model = model, dataset = dataset)]

  if (cache && !is.null(dataset) && !is.null(model)) {
    dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
    saveRDS(eff, cache_file)
  }

  eff
}

load_sfm <- function(dataset) {
  
  if (dataset == "nsduh") {
    sfm <- list(
      X = "race",
      Z = c("age", "sex"),
      W = c("edu", "income"),
      Y = "mj_monthly"
    )
  } else if (dataset == "brfss") {
    sfm <- list(
      X = "race",
      Z = c("age_group", "sex"),
      W = c("education", "income", "bmi", "exercise_monthly"),
      Y = "diabetes"
    )
  } else if (dataset == "census_income") {
    sfm <- list(
      X = "sex",
      Z = c("age_group", "race", "economic_region"),
      W = c("education", "hours_worked", "employer"),
      Y = "salary_group"
    )
  } else if (dataset == "census_doctor") {
    sfm <- list(
      X = "sex",
      Z = c("age_group", "race", "economic_region"),
      W = c("marital", "children", "family_size", "hours_worked"),
      Y = "doctor"
    )
  } else if (dataset == "census_surgeon") {
    sfm <- list(
      X = "sex",
      Z = c("age_group", "race", "economic_region"),
      W = c("marital", "children", "family_size", "hours_worked"),
      Y = "surgeon"
    )
  } else {
    stop("Unknown dataset: ", dataset)
  }
  
  sfm
}
