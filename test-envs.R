
root <- rprojroot::find_root(rprojroot::has_file(".gitignore"))
invisible(lapply(list.files(file.path(root, "r"), full.names = TRUE), 
                 source))

# generate the data from the two environments
sims <- c()

for (seed in seq_len(5)) {
  
  set.seed(seed)
  envs <- c("A", "C")
  
  data1 <- gen_from_scm_modular(envs[1], envs[1], envs[1], 10^4)
  data2 <- gen_from_scm_modular(envs[2], envs[2], envs[2], 10^4)
  
  env_data <- rbind(
    cbind(data1$data, E = 0),
    cbind(data2$data, E = 1)
  )
  
  po_est <- one_step_debias(env_data, X = "X", Z = data1$mapping$Z,
                            W = data1$mapping$W, Y = "Y", E = "E", 
                            return_pos = TRUE)
  
  i <- 0
  res <- c()
  for (xz in c(0, 1)) for (xw in c(0, 1)) for (xy in c(0, 1)) {
    
    for (ez in c(0, 1)) for (ew in c(0, 1)) for (ey in c(0, 1)) {
      
      i <- i+1
      gt <- compute_PO_modular(envs[ez+1], envs[ew+1], envs[ey+1], xz, xw, xy)
      
      po <- po_est[[xz+1]][[ez+1]][[xw+1]][[ew+1]][[xy+1]][[ey+1]]
      
      res <- rbind(
        res,
        data.frame(
          index = i, gt = gt, est = po[1], sd = po[2], zval = abs(gt - po[1]) / po[2],
          xz = xz, xw = xw, xy = xy, ez = ez, ew = ew, ey = ey
        )
      )
    }
  }
  
  res <- as.data.table(res)
  res[, seed := seed]
  sims <- rbind(sims, res) 
}

sims[, cross_env := (ez != ew | ez != ey)]
ggplot(sims, aes(x = index, y = zval, color = cross_env)) + 
  geom_point() + theme_bw() + geom_hline(yintercept = 2, color = "red")


sims[, mean(zval), by = "index"][V1 > 2]$index

hard_ab <- c(5, 6, 7, 10, 13, 14, 15, 19, 20, 21, 22, 23, 26, 27, 28, 29, 
             30, 31, 34, 37, 38, 39, 42, 46, 47, 50, 52, 53, 55, 58, 60, 62, 
             63)

intersect(sims[, mean(zval), by = "index"][V1 > 2]$index, hard_ab)

# ggplot(res, aes(x = index, y = est)) +
#   geom_errorbar(aes(ymin = est - 1.96 * sd, ymax = est + 1.96 * sd)) +
#   geom_point(aes(y = gt), color = "red") + theme_bw()
#     
# 
# ggplot(res[cross_env == TRUE], aes(x = zval, fill = nested_mean)) + 
#   geom_density() + theme_bw()

# checking the P(e | x, z, w) values (roughly)

e_logreg <- glm(E ~ . - Y, data = env_data)

env_data_x1 <- env_data_x0 <- env_data
env_data_x1$X <- 1
env_data_x0$X <- 0

l_pe <- list(
  expit(predict(e_logreg, env_data_x0)),
  expit(predict(e_logreg, env_data_x1))
)

plot(l_pe[[1 + 0]], cfit$pe_xzw[[1 + 0]][[2]], pch = 19)
abline(0, 1)

plot(l_pe[[1 + 1]], cfit$pe_xzw[[1 + 1]][[2]], pch = 19)
abline(0, 1)
