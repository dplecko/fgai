
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), source))

datasets <- DATASETS
models   <- MODELS

# --- estimate everything across all datasets --------------------------------

eff_all <- rbindlist(lapply(datasets, function(ds) {
  sfm <- load_sfm(ds)
  rbindlist(lapply(models, function(m) {
    message(f("[{ds}] Processing: {m}"))
    df_lst <- load_model_data(ds, m)
    estimate_within(df_lst, sfm$X, sfm$Z, sfm$W, sfm$Y,
                    dataset = ds, model = m)
  }))
}))

scores_all <- score_eff(eff_all)

# --- per-dataset tables (appendix) ------------------------------------------

for (ds in datasets) {
  write_stereotype_latex(
    scores_all[dataset == ds],
    f("results/stereotype-table-{ds}.tex"),
    caption = f("Stereotype analysis: {ds}."),
    label = f("tab:stereotype-{ds}"),
    include_advantage = TRUE
  )
}

# --- aggregate table (main text) --------------------------------------------

write_stereotype_latex(
  copy(scores_all),
  "results/stereo.tex",
  caption = "Stereotype analysis summary (all datasets).",
  label = "tab:stereotype-all"
)

# --- real-world discrimination table ----------------------------------------

write_world_latex(
  eff_all,
  "results/world.tex",
  caption = "Real-world discrimination across datasets with 95\\% confidence intervals.",
  label = "tab:world"
)
