# --- 9D vector extraction ----------------------------------------------------

#' Wide matrix of 9D vectors: rows = models, cols = ce_stage
to_9d_matrix <- function(eff) {
  wide <- dcast(eff[stage != "world" & ce %in% c("de", "ie", "se")],
                model ~ dataset + ce + stage, value.var = "value")
  mat <- as.matrix(wide[, -"model"])
  rownames(mat) <- wide$model
  mat
}

# --- distance functions ------------------------------------------------------

#' L1 distance matrix
dist_l1 <- function(mat) {
  as.dist(apply(mat, 1, function(x) colSums(abs(t(mat) - x))))
}

#' Sign-agreement distance: proportion of components with different signs
#' Returns a dist object in [0, 1]; 0 = identical sign pattern
dist_sign <- function(mat) {
  sgn <- sign(mat)
  n <- nrow(sgn)
  d <- matrix(0, n, n, dimnames = list(rownames(mat), rownames(mat)))
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      d[i, j] <- d[j, i] <- mean(sgn[i, ] != sgn[j, ])
    }
  }
  as.dist(d)
}

# --- MDS ---------------------------------------------------------------------

#' Classical MDS into 2D
run_mds <- function(d, k = 2) {
  fit <- cmdscale(d, k = k)
  data.table(
    model = rownames(fit),
    dim1 = fit[, 1],
    dim2 = fit[, 2]
  )
}

# --- plotting ----------------------------------------------------------------

plot_mds <- function(mds_dt, title = "Model Similarity (MDS)") {
  
  ggplot(mds_dt, aes(x = dim1, y = dim2, label = model)) +
    geom_point(size = 3) +
    ggrepel::geom_text_repel(size = 3.5, max.overlaps = 20) +
    labs(x = "Dimension 1", y = "Dimension 2", title = title) +
    theme_bw()
}