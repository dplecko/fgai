# =============================================================================
# fgai-helpers.R — shared core for all FGAI analyses
#
# Pipeline:  load_model_data → estimate_within → extract_stage_effects
#
# Everything downstream (9D vectors, waterfall, scoring, similarity)
# reads from the 12-row stage effects table.
# =============================================================================

# --- data loading -----------------------------------------------------------

#' Load the 4 environment datasets for a given model
#' Returns a named list keyed by env suffixes: "", "XZ", "XZW", "XZWY"
load_model_data <- function(dataset, model) {
  
  X_var <- load_sfm(dataset)$X
  if (is.element(dataset, c("brfss", "nsduh"))) {
    
    X_keep <- c("Black", "White")
    X_ref <- "Black"
  } else if (grepl("census", dataset)) {
    
    X_keep <- c("female", "male")
    X_ref <- "female"
  }
  
  envs <- c("", "XZ", "XZW", "XZWY")
  df_lst <- lapply(
    envs,
    function(x) {
      fl <- paste0(paste0(c(dataset, model, x), collapse = "_"), ".parquet")
      df <- as.data.frame(read_parquet(file.path("data", fl)))
      df <- df[df[[X_var]] %in% X_keep, ]
      df[[X_var]] <- as.integer(df[[X_var]] == X_ref)
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
#' Results cached to results/cache/{dataset}_{model}.rds
#'
#' @return data.table with 12 rows: stage × ce, columns: stage, ce, value, sd
estimate_within <- function(df_lst, X, Z, W, Y,
                            dataset = NULL, model = NULL,
                            eps_trim = 0.001, cache = TRUE) {

  # caching
  if (cache && !is.null(dataset) && !is.null(model)) {
    cache_dir <- "results/cache"
    cache_file <- file.path(cache_dir, f("{dataset}_{model}.rds"))
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
