

#' @importFrom xgboost xgb.DMatrix xgb.cv xgb.train
cv_xgb_env <- function(df, y, weights = NULL, ...) {
  
  dtrain <- xgb.DMatrix(data = as.matrix(df), label = y, weight = weights)
  
  binary <- all(y %in% c(0, 1))
  if (binary) {
    
    params <- list(objective = "binary:logistic", eval_metric = "logloss")
  } else {
    
    params <- list(objective = "reg:squarederror", eval_metric = "rmse")
  }
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 10,
    prediction = TRUE,
    verbose = FALSE, ...
  )
  
  xgb <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = cv$best_iteration,
    verbose = FALSE, ...
  )
  attr(xgb, "binary") <- binary
  
  xgb
}

pred_xgb_env <- function(xgb, df_test, intervention = NULL, X = list("X", "E")) {
  
  for (i in seq_along(intervention)) {
    
    df_test[[ X[i] ]] <- intervention[i]
  }
  
  predict(xgb, as.matrix(df_test))
}

cross_fit_env <- function(data, X, Z, W, Y, E = NULL, nested_mean = "refit", 
                          log_risk = FALSE, ...) {
  
  if (length(Z) == 0 & length(W) == 0) {
    
    px <- rep(mean(data[[X]]), nrow(data))
    return(list(px_z = list(1-px, px), px_zw = list(1-px, px)))
  }
  
  # split into K folds
  n <- nrow(data)
  K <- 10
  folds <- sample(x = rep(1:K, each = ceiling(n / K)))[seq_len(n)]
  
  # elements to be filled
  px_zw <- px_z <- list(rep(NA, n), rep(NA, n))
  
  y_xezw <- y_xez <- ey_nest <- pe_xz <- pe_xzw <-
    list(list(rep(NA, n), rep(NA, n)), list(rep(NA, n), rep(NA, n)))
  
  ey_nest <- list(list(ey_nest, ey_nest), list(ey_nest, ey_nest))
  
  # x, y data
  y <- data[[Y]]
  x <- data[[X]]
  e <- data[[E]]
  
  # cross-fit
  for (i in seq_len(K)) {
    
    #' * fold splitting * split into dev, val, tst
    tst <- folds == i
    dev <- folds %in% setdiff(seq_len(K), i)[1:6]
    val <- folds %in% setdiff(seq_len(K), i)[7:9]
    
    #' * model fitting * develop models on dev
    if (length(Z) > 0) {
      
      # X / E models
      mod_x_z <- cv_xgb_env(data[dev, Z], data[dev, X], ...)
      mod_e_xz <- cv_xgb_env(data[dev, c(X, Z)], data[dev, E], ...)
      
      # Y model
      mod_y_xez <- cv_xgb_env(data[dev, c(X, E, Z)], data[dev, Y], ...)
    }
    
    if (length(W) > 0) {
      
      mod_x_zw <- cv_xgb_env(data[dev, c(Z, W)], data[dev, X], ...)
      mod_e_xzw <- cv_xgb_env(data[dev, c(X, Z, W)], data[dev, E], ...)
      
      mod_y_xezw <- cv_xgb_env(data[dev, c(X, E, Z, W)], data[dev, Y], ...)
    } else {
      
      # inherit from Z if W empty
      mod_x_zw <- mod_x_z
      mod_e_xzw <- mod_e_xz
      mod_y_xezw <- mod_y_xez
    }
    
    # get the val set predictions for Y | X, Z, W (for nested means)
    y_xzw_val <- list(
      list(
        pred_xgb_env(mod_y_xezw, data[val, c(X, E, Z, W)], intervention = c(0, 0), X = c(X, E)),
        pred_xgb_env(mod_y_xezw, data[val, c(X, E, Z, W)], intervention = c(0, 1), X = c(X, E))
      ),
      list(
        pred_xgb_env(mod_y_xezw, data[val, c(X, E, Z, W)], intervention = c(1, 0), X = c(X, E)),
        pred_xgb_env(mod_y_xezw, data[val, c(X, E, Z, W)], intervention = c(1, 1), X = c(X, E))
      )
    )
    
    #' * get the test set values * 
    
    # P(X | Z, W)
    px_zw_tst <- pred_xgb_env(mod_x_zw, data[tst, c(Z, W)])
    px_zw[[1 + 0]][tst] <- 1 - px_zw_tst
    px_zw[[1 + 1]][tst] <- px_zw_tst
    
    # P(E | X, Z, W)
    pe_x0zw_tst <- pred_xgb_env(mod_e_xzw, data[tst, c(X, Z, W)], intervention = 0, X = X)
    pe_x1zw_tst <- pred_xgb_env(mod_e_xzw, data[tst, c(X, Z, W)], intervention = 1, X = X)
    pe_xzw[[1 + 0]][[1 + 0]][tst] <- 1 - pe_x0zw_tst
    pe_xzw[[1 + 0]][[1 + 1]][tst] <- pe_x0zw_tst
    pe_xzw[[1 + 1]][[1 + 0]][tst] <- 1 - pe_x1zw_tst
    pe_xzw[[1 + 1]][[1 + 1]][tst] <- pe_x1zw_tst
    
    # P(X | Z)
    px_z_tst <- pred_xgb_env(mod_x_z, data[tst, Z])
    px_z[[1 + 0]][tst] <- 1 - px_z_tst
    px_z[[1 + 1]][tst] <- px_z_tst
    
    # P(E | X, Z)
    pe_x0z_tst <- pred_xgb_env(mod_e_xz, data[tst, c(X, Z)], intervention = 0, X = X)
    pe_x1z_tst <- pred_xgb_env(mod_e_xz, data[tst, c(X, Z)], intervention = 1, X = X)
    pe_xz[[1 + 0]][[1 + 0]][tst] <- 1 - pe_x0z_tst
    pe_xz[[1 + 0]][[1 + 1]][tst] <- pe_x0z_tst
    pe_xz[[1 + 1]][[1 + 0]][tst] <- 1 - pe_x1z_tst
    pe_xz[[1 + 1]][[1 + 1]][tst] <- pe_x1z_tst
    
    
    y_xezw[[1 + 0]][[1 + 0]][tst] <- pred_xgb_env(mod_y_xezw, data[tst, c(X, E, Z, W)],
                                                  intervention = c(0, 0), X = c(X, E))
    y_xezw[[1 + 1]][[1 + 0]][tst] <- pred_xgb_env(mod_y_xezw, data[tst, c(X, E, Z, W)],
                                                  intervention = c(1, 0), X = c(X, E))
    y_xezw[[1 + 0]][[1 + 1]][tst] <- pred_xgb_env(mod_y_xezw, data[tst, c(X, E, Z, W)],
                                                  intervention = c(0, 1), X = c(X, E))
    y_xezw[[1 + 1]][[1 + 1]][tst] <- pred_xgb_env(mod_y_xezw, data[tst, c(X, E, Z, W)],
                                                  intervention = c(1, 1), X = c(X, E))
    
    # nested means are not needed if either Z or W are empty
    if (length(Z) == 0 || length(W) == 0) next
    
    for (xw in c(0, 1)) for (xy in c(0, 1)) for (ey in c(0, 1)) for (ew in c(0, 1)) {
      
      if (xw == xy & ew == ey) {
        
        # get predictions from the Y | X, E, Z model!
        ey_nest[[xw+1]][[ew+1]][[xy+1]][[ey+1]][tst] <-
          pred_xgb_env(mod_y_xez, data[tst, c(X, E, Z)], 
                       intervention = c(xw, ew), X = c(X, E))
      } else {
        
        # re-fitting needed
        y_tilde <- pred_xgb_env(mod_y_xezw, data[val, c(X, E, Z, W)],
                                intervention = c(xy, ey), X = c(X, E))
        
        mod_nested <- cv_xgb_env(data[val, c(X, E, Z)], y_tilde, ...)
        ey_nest[[xw+1]][[ew+1]][[xy+1]][[ey+1]][tst] <- 
          pred_xgb_env(mod_nested, data[tst, c(X, E, Z)], 
                       intervention = c(xw, ew), X = c(X, E))
      }
    }
  }
  
  list(
    y_xezw = y_xezw,
    ey_nest = ey_nest,
    px_z = px_z,
    px_zw = px_zw,
    pe_xzw = pe_xzw,
    pe_xz = pe_xz
  )
}

pso_diff_env <- function(cfit, data, X, Z, W, Y, E, ...) {
  
  n <- nrow(data)
  
  # un-nest the cfit object
  y_xezw <- cfit$y_xezw
  ey_nest <- cfit$ey_nest
  px_z <- cfit$px_z
  px_zw <- cfit$px_zw
  pe_xz <- cfit$pe_xz
  pe_xzw <- cfit$pe_xzw
  
  
  # get data and psos
  y <- data[[Y]]
  x <- data[[X]]
  e <- data[[E]]
  
  pso <- list()
  for (i in 1:6) pso <- list(pso, pso)
  
  for (xz in c(0, 1)) for (xw in c(0, 1)) for (xy in c(0, 1))
    for (ez in c(0, 1)) for (ew in c(0, 1)) for (ey in c(0, 1)) {
    
    pso[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]] <- rep(NA, n)
  }
  
  for (xz in c(0, 1)) for (xw in c(0, 1)) for (xy in c(0, 1))
    for (ez in c(0, 1)) for (ew in c(0, 1)) for (ey in c(0, 1)) {
    
    pso[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]] <-
      
      # Term T1
      (x == xy & e == ey) /  mean(x == xz & e == ez) * 
      px_z[[xz+1]] * pe_xz[[xz+1]][[ez+1]] / (px_z[[xw+1]] * pe_xz[[xw+1]][[ew+1]]) *
      px_zw[[xw+1]] * pe_xzw[[xw+1]][[ew+1]] / (px_zw[[xy+1]] * pe_xzw[[xy+1]][[ey+1]]) *
      (y - y_xezw[[xy+1]][[ey+1]]) + 
      
      # Term T2
      (x == xw & e == ew) / mean(x == xz & e == ez) * 
      px_z[[xz+1]] * pe_xz[[xz+1]][[ez+1]] / ( px_z[[xw+1]] * pe_xz[[xw+1]][[ew+1]] ) *
      (y_xezw[[xy+1]][[ey+1]] - ey_nest[[xw+1]][[ew+1]][[xy+1]][[ey+1]]) +
      
      # Term T3
      (x == xz & e == ez) / mean(x == xz & e == ez) * 
        ey_nest[[xw+1]][[ew+1]][[xy+1]][[ey+1]]
  }
  
  pso
}

measure_spec_env <- function() {
  
  ia <- list(
    # first order differences - world
    tv_w = list(
      sgn = c(1, -1),
      spc = list(c(1, 0, 1, 0, 1, 0), c(0, 0, 0, 0, 0, 0)),
      ia = "tv-world"
    ),
    ctfde_w = list(
      sgn = c(1, -1),
      spc = list(c(0, 0, 0, 0, 1, 0), c(0, 0, 0, 0, 0, 0)),
      ia = "ctfde-world"
    ),
    ctfie_w = list(
      sgn = c(1, -1),
      spc = list(c(0, 0, 0, 0, 1, 0), c(0, 0, 1, 0, 1, 0)),
      ia = "ctfie-world"
    ),
    ctfse_w = list(
      sgn = c(1, -1),
      spc = list(c(0, 0, 1, 0, 1, 0), c(1, 0, 1, 0, 1, 0)),
      ia = "ctfse-world"
    ),
    
    # first order differences - model
    tv_m = list(
      sgn = c(1, -1),
      spc = list(c(1, 1, 1, 1, 1, 1), c(0, 1, 0, 1, 0, 1)),
      ia = "tv-model"
    ),
    ctfde_m = list(
      sgn = c(1, -1),
      spc = list(c(0, 1, 0, 1, 1, 1), c(0, 1, 0, 1, 0, 1)),
      ia = "ctfde-model"
    ),
    ctfie_m = list(
      sgn = c(1, -1),
      spc = list(c(0, 1, 0, 1, 1, 1), c(0, 1, 1, 1, 1, 1)),
      ia = "ctfie-model"
    ),
    ctfse_m = list(
      sgn = c(1, -1),
      spc = list(c(0, 1, 1, 1, 1, 1), c(1, 1, 1, 1, 1, 1)),
      ia = "ctfse-model"
    ),
    
    # second order differences - model vs. world
    # direct decomposition
    delta_de_fy = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 0, 0, 1, 0), c(0, 0, 0, 0, 0, 0),
        c(0, 0, 0, 0, 1, 1), c(0, 0, 0, 0, 0, 1)
      ),
      ia = "delta-de-fy"
    ),
    delta_de_fw = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 0, 0, 1, 1), c(0, 0, 0, 0, 0, 1),
        c(0, 0, 0, 1, 1, 1), c(0, 0, 0, 1, 0, 1)
      ),
      ia = "delta-de-fw"
    ),
    delta_de_fz = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 0, 1, 1, 1), c(0, 0, 0, 1, 0, 1),
        c(0, 1, 0, 1, 1, 1), c(0, 1, 0, 1, 0, 1)
      ),
      ia = "delta-de-fz"
    ),
    
    # indirect decomposition
    delta_ie_fy = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 0, 0, 1, 0), c(0, 0, 1, 0, 1, 0),
        c(0, 0, 0, 0, 1, 1), c(0, 0, 1, 0, 1, 1)
      ),
      ia = "delta-ie-fy"
    ),
    delta_ie_fw = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 0, 0, 1, 1), c(0, 0, 1, 0, 1, 1),
        c(0, 0, 0, 1, 1, 1), c(0, 0, 1, 1, 1, 1)
      ),
      ia = "delta-ie-fw"
    ),
    delta_ie_fz = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 0, 1, 1, 1), c(0, 0, 1, 1, 1, 1),
        c(0, 1, 0, 1, 1, 1), c(0, 1, 1, 1, 1, 1)
      ),
      ia = "delta-ie-fz"
    ),
    
    # spurious decomposition
    delta_se_fy = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 1, 0, 1, 0), c(1, 0, 1, 0, 1, 0),
        c(0, 0, 1, 0, 1, 1), c(1, 0, 1, 0, 1, 1)
      ),
      ia = "delta-se-fy"
    ),
    delta_se_fw = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 1, 0, 1, 1), c(1, 0, 1, 0, 1, 1),
        c(0, 0, 1, 1, 1, 1), c(1, 0, 1, 1, 1, 1)
      ),
      ia = "delta-se-fw"
    ),
    delta_se_fz = list(
      sgn = c(1, -1, -1, 1),
      spc = list(
        c(0, 0, 1, 1, 1, 1), c(1, 0, 1, 1, 1, 1),
        c(0, 1, 1, 1, 1, 1), c(1, 1, 1, 1, 1, 1)
      ),
      ia = "delta-se-fz"
    )
  )
}

one_step_debias_env <- function(data, X, Z, W, Y, E, nested_mean = "refit",
                                eps_trim = 0, return_pos = FALSE, ...) {
  
  cfit <- cross_fit_env(data, X, Z, W, Y, E, nested_mean, log_risk, ...)
  pso <- pso_diff_env(cfit, data, X, Z, W, Y, E, ...)
  
  # get extreme propensity weights
  extrm_pxez <- extrm_pxezw <- rep(FALSE, nrow(data))
  for (xy in c(0, 1)) for (ey in c(0, 1)) {
    
    pxe_z <- cfit$px_z[[1+xy]] * cfit$pe_xz[[1+xy]][[1+ey]]
    extrm_pxez <- extrm_pxez | (pxe_z < eps_trim) | (1 - pxe_z < eps_trim)
    
    pxe_zw <- cfit$px_zw[[1+xy]] * cfit$pe_xzw[[1+xy]][[1+ey]]
    extrm_pxezw <- extrm_pxezw | (pxe_zw < eps_trim) | (1 - pxe_zw < eps_trim)
  }
  extrm_idx <- extrm_pxez | extrm_pxezw
  
  # report if large number of propensity weights below specified threshold
  if (mean(extrm_idx) > 0.02) {
    message(round(100 * mean(extrm_idx), 2),
            "% of extreme P(x, e | z) or P(x, e | z, w) probabilities at threshold",
            " = ", eps_trim, ".\n",
            "Reported results are for the overlap population. ",
            "Consider investigating overlap issues.")
  }
  
  # trim population to extreme weights
  for (xz in c(0, 1)) for (xw in c(0, 1)) for (xy in c(0, 1))
    for (ez in c(0, 1)) for (ew in c(0, 1)) for (ey in c(0, 1))
    pso[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]][extrm_idx] <- NA
  
  # get specification of measures to be reported
  
  if (return_pos) {
    
    for (xz in c(0, 1)) for (xw in c(0, 1)) for (xy in c(0, 1))
      for (ez in c(0, 1)) for (ew in c(0, 1)) for (ey in c(0, 1)) {
        
        pso_curr <- pso[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]]
        pso[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]] <- 
          c(mean(pso_curr, na.rm = TRUE), 
            sqrt(var(pso_curr, na.rm = TRUE) / sum(!is.na(pso_curr))))
      }
    
    return(pso)
  }
  
  ias <- measure_spec_env()
  
  res <- c()
  for (i in seq_along(ias)) {
    
    pseudo_out <- 0
    for (j in seq_along(ias[[i]]$sgn)) {
      
      xz <- ias[[i]]$spc[[j]][1]
      ez <- ias[[i]]$spc[[j]][2]
      xw <- ias[[i]]$spc[[j]][3]
      ew <- ias[[i]]$spc[[j]][4]
      xy <- ias[[i]]$spc[[j]][5]
      ey <- ias[[i]]$spc[[j]][6]
      pseudo_out <- pseudo_out + ias[[i]]$sgn[j] * 
        pso[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]]
    }
    psi_osd <- mean(pseudo_out, na.rm = TRUE)
    dev <- sqrt(var(pseudo_out, na.rm = TRUE) / sum(!is.na(pseudo_out)))
    
    res <- rbind(
      res,
      data.frame(measure = ias[[i]]$ia, value = psi_osd, sd = dev)
    )
  }
  
  # pw <- list(px_z = cfit$px_z, px_zw = cfit$px_zw)
  # attr(res, "pw") <- pw
  res
}
