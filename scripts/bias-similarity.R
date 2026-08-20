
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

datasets <- DATASETS
models   <- MODELS

# --- estimate all models across all datasets --------------------------------

eff_all <- rbindlist(lapply(datasets, function(ds) {
  sfm <- load_sfm(ds)
  rbindlist(lapply(models, function(m) {
    message(f("[{ds}] Processing: {m}"))
    df_lst <- load_model_data(ds, m)
    estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                    dataset = ds, model = m)
  }))
}))

# --- per-dataset MDS (appendix) ---------------------------------------------

# for (ds in datasets) {
#   mat <- to_9d_matrix(eff_all[dataset == ds])
#   mds <- run_mds(dist_l1(mat))
#   mds[, model := MODEL_NAMES[model]]
#   ggsave(f("results/mds-{ds}.png"),
#          plot = plot_mds(mds, title = f("Model Similarity ({ds})")),
#          width = 7, height = 5)
# }

# --- joint MDS: concatenate across datasets (main text) ---------------------

mat_all <- to_9d_matrix(eff_all)
mds_all <- run_mds(dist_l1(mat_all))
mds_all[, model := MODEL_NAMES[model]]
ggsave("results/mds-all.png",
       plot = plot_mds(mds_all, title = NULL),
       width = 7, height = 5)

# --- heatmap (upper triangle only) ------------------------------------------

D <- dist_l1(mat_all)
hc <- hclust(D, method = "ward.D2")
ord <- hc$order

dmat <- as.matrix(D)
dmat <- dmat[ord, ord]
rownames(dmat) <- MODEL_NAMES[rownames(dmat)]
colnames(dmat) <- MODEL_NAMES[colnames(dmat)]

# keep only upper triangle (excluding diagonal)
dmat[lower.tri(dmat, diag = TRUE)] <- NA

dt <- as.data.table(reshape2::melt(dmat, na.rm = TRUE))
setnames(dt, c("model1", "model2", "distance"))
dt[, model1 := factor(model1, levels = rownames(dmat))]
dt[, model2 := factor(model2, levels = rev(colnames(dmat)))]   # reverse x-axis

p_heat <- ggplot(dt, aes(model2, model1, fill = distance)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.2f", distance)), size = 3) +
  scale_fill_gradient(low = "#2e7d32", high = "white") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = NULL, y = NULL, fill = "L1 distance")

ggsave("results/heatmap-all.png", plot = p_heat, width = 7, height = 5)

# --- dendrogram -------------------------------------------------------------

hc$labels <- MODEL_NAMES[hc$labels]

p_dendro <- ggdendrogram(hc, rotate = FALSE, theme_dendro = FALSE) +
  labs(x = NULL, y = "L1 distance") +
  theme_bw(base_size = 16) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank())

ggsave("results/dendrogram-all.png", plot = p_dendro,
       width = 7, height = 5)
