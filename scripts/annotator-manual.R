

pair <- "brfss_qwen35_27b" # "nsduh_gemma3_27b" "brfss_qwen35_27b"
d1 <- as.data.frame(read_parquet(f("data/{pair}_commandrp_104b_.parquet")))
d2 <- as.data.frame(read_parquet(f("data/{pair}_qwen25_72b_.parquet")))

sim <- rep(NA, nrow(d1))
for (i in seq_len(nrow(d1))) {
  
  sim[i] <- mean(d1[i, ] == d2[i, ])
}

mean(sim)
which(sim <= 0.5)

sim_mat <- c()
for (i in 1:ncol(d1)) {
  
  sim_mat <- cbind(sim_mat, as.character(d1[[i]]) == as.character(d2[[i]]))
}

colMeans(sim_mat)


idx <- 16

for (idx in which(!sim_mat[, 6])) {
  
  print(d1[idx, ])
  print(d2[idx, ])
  cat(read_parquet(f("data/cache/{pair}__gen.parquet"))$response[idx])
  browser()
}



ann <- as.data.table(read_parquet("data/cache/nsduh_llama3_8b_llama3_70b__ann.parquet"))

samp <- function() {
  
  idx <- sample.int(nrow(ann), 1)
  
  cat("Variable:", ann[idx]$variable, "\n")
  cat(sub("(?s)^.*<story>", "", ann[idx]$prompt, perl = TRUE), "\n\n")
  cat(ann[idx]$response, "\n\n")
  cat("Predicted:", ann[idx]$predicted, "\n")
}

samp()

head(ann)
