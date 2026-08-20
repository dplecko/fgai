
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

# --- config -----------------------------------------------------------------

datasets <- DATASETS
models   <- MODELS
N_PERM   <- 5000

FAMILIES <- list(
  llama    = c("llama3_8b", "llama3_70b"),
  qwen     = c("qwen35_9b", "qwen35_27b"),
  mistral = c("ministral3_8b", "ministral3_14b", "mistral_24b")
)

# build family label vector aligned with `models`
family_labels <- sapply(models, function(m) {
  fam <- names(FAMILIES)[sapply(FAMILIES, function(f) m %in% f)]
  if (length(fam) == 0) m else fam   # singleton gets its own unique label
})

# --- estimate, build matrix, distance ---------------------------------------

eff_all <- rbindlist(lapply(datasets, function(ds) {
  sfm <- load_sfm(ds)
  rbindlist(lapply(models, function(m) {
    df_lst <- load_model_data(ds, m)
    estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                    dataset = ds, model = m)
  }))
}))

mat   <- to_9d_matrix(eff_all)        # rows = models, cols = ce_stage_dataset
mat   <- mat[models, ]                # ensure ordering matches family_labels
D_mat <- as.matrix(dist_l1(mat))      # full n×n distance matrix
diag(D_mat) <- Inf                    # exclude self

# --- statistic functions ----------------------------------------------------

#' NN hit count: # of models whose nearest neighbor shares family
#' (singletons can never hit since their family has size 1)
nn_hits <- function(D, labels) {
  fam_size <- table(labels)[labels]
  nn_idx   <- apply(D, 1, which.min)
  sum(labels == labels[nn_idx] & fam_size > 1)
}

mean_within_l1 <- function(D, labels) {
  fam_size <- table(labels)[labels]
  has_sib  <- fam_size > 1
  ds <- numeric(0)
  for (i in which(has_sib)) {
    sibs <- which(labels == labels[i] & seq_along(labels) != i)
    ds <- c(ds, D[i, sibs])
  }
  mean(ds)
}

# --- observed ---------------------------------------------------------------

obs_hits <- nn_hits(D_mat, family_labels)
obs_l1  <- mean_within_l1(D_mat, family_labels)

# --- permutation -------------------------------------------------------------

set.seed(2026)

null_l1 <- replicate(N_PERM, mean_within_l1(D_mat, sample(family_labels)))

null_hits <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  perm <- sample(family_labels)
  null_hits[i] <- nn_hits(D_mat, perm)
}

# --- p-values ---------------------------------------------------------------

# L1 distance
p_l1    <- mean(null_l1 <= obs_l1)

# NN hits: right-tail (more hits = stronger family signal)
p_hits <- mean(null_hits >= obs_hits)

# --- report -----------------------------------------------------------------

cat("=== Permutation Test (N =", N_PERM, ") ===\n\n")

cat("L1 Distance\n")
cat(f("  observed: {obs_l1}\n"))
cat(f("  null mean: {round(mean(null_l1), 2)}\n"))
cat(f("  p-value (left-tail): {round(p_l1, 4)}\n\n"))

cat("NN Hit Count\n")
cat(f("  observed: {obs_hits}\n"))
cat(f("  null mean: {round(mean(null_hits), 2)}\n"))
cat(f("  p-value (right-tail): {round(p_hits, 4)}\n\n"))
