library(GpGp)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  scene <- args[1]
  N <- as.integer(args[2])
  d <- as.integer(args[3])
} else {
  scene <- "1Dexponential_heteroskedasticity"
  N <- 100
  d <- 1
}
m_vec <- seq(from = 3, to = 30, by = 3)

covname <- "matern15_isotropic"
covparm_init <- c(1.0, 0.1, 0.1)
locs_train <- as.matrix(read.csv(paste0("data/", scene, "/0/train.csv"),
  header = FALSE
)[, 1:d, drop = FALSE])
locs_test <- as.matrix(read.csv(paste0("data/", scene, "/0/test.csv"),
  header = FALSE
)[, 1:d, drop = FALSE])
y_train <- matrix(0, nrow(locs_train), N)
y_test <- matrix(0, nrow(locs_test), N)
for (k in 1:N) {
  data <- read.csv(paste("data", scene, k - 1, "train.csv", sep = "/"),
    header = FALSE
  )
  y_train[, k] <- data[, d+1]
  data <- read.csv(paste("data", scene, k - 1, "test.csv", sep = "/"),
    header = FALSE
  )
  y_test[, k] <- data[, d+1]
}

mse <- numeric(length(m_vec))
ind <- 1
for (m in m_vec) {
  NN <- GpGp::find_ordered_nn(locs_train, m)
  nll_func <- function(logcovparm) {
    covparm <- exp(logcovparm)
    llk <- apply(y_train, MARGIN = 2, FUN = function(y) {
      GpGp::vecchia_meanzero_loglik(
        covparm, covname, y, locs_train, NN
      )
    })
    -mean(unlist(llk))
  }
  optim_rslt <- optim(log(covparm_init), fn = nll_func)
  covparm <- exp(optim_rslt$par)
  mse_func <- function(covparm) {
    y_pred <- apply(y_train, 2, FUN = function(y) {
      GpGp::predictions(
        locs_pred = locs_test, X_pred = rep(0, nrow(locs_test)),
        y_obs = y, locs_obs = locs_train,
        X_obs = rep(0, nrow(locs_train)), beta = 0, covparms = covparm,
        covfun_name = covname, m = m, reorder = TRUE
      )
    })
    mean((y_test - y_pred)^2)
  }
  mse[ind] <- mse_func(covparm)
  cat(paste(m, mse[ind], sep = ","), "\n")
  ind <- ind + 1
}













