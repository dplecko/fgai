
## --- shared X generator (unchanged) --------------------
gen_X <- function(Z, option, n) {
  if (option == "random") {
    return(rbinom(n, 1, prob = plogis(Z %*% c(0.3, -0.2, 0.5) + 0.2 * (Z[,1]^2))))
  } else if (option == 0) {
    return(rep(0, n))
  } else if (option == 1) {
    return(rep(1, n))
  } else {
    stop("X option must be 'random', 0, or 1")
  }
}

## --- Z mechanisms --------------------------------------
mech_Z <- list(
  A = function(n) {
    
    pi <- 0.7
    cmp <- rbinom(n, 1, pi)
    
    Sigma <- matrix(
      c(
        1, 1/4, 1/5,
        1/4, 1, -1/4,
        1/5, -1/4, 1
      ), ncol=3, byrow = TRUE
    )
    Z1 <- MASS::mvrnorm(n, c(0.5, 1, 0.5), Sigma = Sigma)
    Z0 <- matrix(rnorm(n * 3), n, 3)
    
    Z1 * cmp + Z0 * (1 - cmp)
  },
  B = function(n) cbind(rnorm(n, 1, 1.5), rnorm(n, -1, 1.5), rnorm(n, 1, 1.5)),
  C = function(n) {
    
    pi <- 0.3
    cmp <- rbinom(n, 1, pi)
    
    Sigma <- matrix(
      c(
        1, 1/4, 1/5,
        1/4, 1, -1/4,
        1/5, -1/4, 1
      ), ncol=3, byrow = TRUE
    )
    Z1 <- MASS::mvrnorm(n, c(1, 1, 1), Sigma = Sigma)
    Z0 <- matrix(rnorm(n * 3), n, 3)
    
    Z1 * cmp + Z0 * (1 - cmp)
  },
  D = function(n) matrix(rnorm(n * 3), n, 3),
  E = function(n) matrix(rnorm(n * 3), n, 3)
)


w_mech1 <- function(Z, X_W) {
  n <- nrow(Z)
  Z %*% matrix(c(0.4, 0.1, -0.3,
                 0.2, -0.1, 0.3,
                 0.3, -0.2, 0.1), 3, 3) +
    X_W * matrix(c(0.5, 0.4, 0.3), n, 3, byrow = TRUE) +
    matrix(rnorm(n * 3), n, 3)
}

w_mech2 <- function(Z, X_W) {
  n <- nrow(Z)
  matrix(c((Z[,1]^2) * 0.3,
           Z[,2] * 0.5,
           X_W * 0.4), n, 3) +
    matrix(rnorm(n * 3), n, 3)
}

## --- W mechanisms: W = f_W^env(Z, X_W) + noise ---------
mech_W <- list(
  A = function(Z, X_W) {
    
    pi <- 0.3
    cmp <- rbinom(nrow(Z), 1, pi)
    W0 <- w_mech1(Z, X_W)
    W1 <- w_mech2(Z, X_W)
    
    W0 * cmp + W1 * (1-cmp)
  },
  B = function(Z, X_W) {
    n <- nrow(Z)
    Z %*% matrix(c(0.3, -0.5, 0.2,
                   -0.1, 0.3, 0.1,
                   0.2, 0.2, -0.3), 3, 3) +
      X_W * matrix(c(0.2, 0.1, 0.4), n, 3, byrow = TRUE) +
      matrix(rnorm(n * 3), n, 3)
  },
  C = function(Z, X_W) {
    
    pi <- 0.7
    cmp <- rbinom(nrow(Z), 1, pi)
    W0 <- w_mech1(Z, X_W)
    W1 <- w_mech2(Z, X_W)
    
    W0 * cmp + W1 * (1-cmp)
  },
  D = function(Z, X_W) {
    n <- nrow(Z)
    Z %*% matrix(c(0.4, 0.1, -0.3,
                   0.2, -0.1, 0.3,
                   0.3, -0.2, 0.1), 3, 3) +
      X_W * matrix(c(0.5, 0.4, 0.3), n, 3, byrow = TRUE) +
      matrix(rnorm(n * 3), n, 3)
  },
  E = function(Z, X_W) {
    n <- nrow(Z)
    matrix(c((Z[,1]^2) * 0.3,
             Z[,2] * 0.5,
             X_W * 0.4), n, 3) +
      matrix(rnorm(n * 3), n, 3)
  }
)

## --- Y mechanisms: Y = f_Y^env(Z, W, X_Y) --------------
## Return list(Y = ..., prob_y = NULL) so binary cases later fit same API
mech_Y <- list(
  A = function(Z, W, X_Y) {
    n <- nrow(W)
    Y <- W %*% c(0.5, 0.4, 0.3) +
      Z %*% c(0.2, 0.1, 0.4) +
      X_Y * 0.7 +
      X_Y * W[,1] * 0.2 +
      rnorm(n)
    list(Y = as.numeric(Y), prob_y = NULL)
  },
  B = function(Z, W, X_Y) {
    n <- nrow(W)
    Y <- W %*% c(0.4, 0.3, 0.2) +
      Z %*% c(0.1, 0.3, 0.2) +
      X_Y * 0.4 + rnorm(n) +
      (W %*% c(0.1, -0.3, -0.3)) * (Z %*% c(0.1, -0.2, 0.2))
    list(Y = as.numeric(Y), prob_y = NULL)
  },
  C = function(Z, W, X_Y) {
    n <- nrow(W)
    Y <- W %*% c(0.3, 0.2, 0.1) +
      Z %*% c(0.2, 0.1, 0.3) +
      rnorm(n)
    list(Y = as.numeric(Y), prob_y = NULL)
  },
  D = function(Z, W, X_Y) {
    n <- nrow(W)
    Y <- W %*% c(0.5, 0.4, 0.3) +
      Z %*% c(0.2, 0.1, 0.4) +
      X_Y * Z[,1] * 0.3 +
      rnorm(n)
    list(Y = as.numeric(Y), prob_y = NULL)
  },
  E = function(Z, W, X_Y) {
    n <- nrow(W)
    Y <- W %*% c(0.4, 0.3, 0.2) +
      Z %*% c(0.2, 0.1, 0.3) +
      X_Y * Z[,1] * W[,3] * 0.5 +
      Z[,2] * W[,3] * (-0.4) +
      X_Y * Z[,3] +
      rnorm(n)
    list(Y = as.numeric(Y), prob_y = NULL)
  }
)

gen_from_scm_modular <- function(ez, ew, ey, n,
                                 X_to_W = "random",
                                 X_to_Y = "random") {
  if (!ez %in% names(mech_Z)) stop("Unknown ez")
  if (!ew %in% names(mech_W)) stop("Unknown ew")
  if (!ey %in% names(mech_Y)) stop("Unknown ey")
  
  ## Z mechanism
  Z <- mech_Z[[ez]](n)
  
  ## "real" X
  X <- gen_X(Z, "random", n)
  
  ## pathway-specific X's
  X_W <- if (X_to_W == "random") X else gen_X(Z, X_to_W, n)
  X_Y <- if (X_to_Y == "random") X else gen_X(Z, X_to_Y, n)
  
  ## W and Y mechanisms
  W <- mech_W[[ew]](Z, X_W)
  y_out <- mech_Y[[ey]](Z, W, X_Y)
  Y <- y_out$Y
  prob_y <- y_out$prob_y
  
  ## names/mapping as in your original code
  col_nms <- function(lab, A)
    if (is.null(A)) NULL else paste0(lab, seq_len(ncol(A)))
  
  nms <- c("X", col_nms("Z", Z), col_nms("W", W), "Y")
  data <- as.data.frame(cbind(X, Z, W, Y))
  names(data) <- nms
  
  ret <- list(
    data = data,
    mapping = list(X = "X",
                   Z = col_nms("Z", Z),
                   W = col_nms("W", W),
                   Y = "Y"),
    env = list(ez = ez, ew = ew, ey = ey)
  )
  if (!is.null(prob_y)) ret$prob_y <- prob_y
  ret
}

compute_PO_modular <- function(ez, ew, ey, xz, xw, xy, n = 1e5) {
  gen <- gen_from_scm_modular(ez, ew, ey,
                              n = n,
                              X_to_W = xw,
                              X_to_Y = xy)
  dat <- gen$data
  
  if (!is.null(gen$prob_y)) dat$Y <- gen$prob_y
  
  ## Z-empty special cases would go here for H/J;
  ## for A–E we just condition on X as before:\
  mean(dat[dat$X %in% xz, "Y"])
}
