library(GpGp)
library(lhs)

set.seed(123)

locs <- lhs::randomLHS(1e4, 2)
y <- GpGp::fast_Gp_sim(c(1, 0.1, 0.0001), "matern15_isotropic", locs, m = 30)
ind <- sample(1:1e4, 1e4)
ind_train <- ind[1:6000]
ind_valid <- ind[6001:8000]
ind_test <- ind[8001:10000]

for (data_type in c("train", "valid", "test")) {
  ind_tmp <- get(paste0("ind_", data_type))
  dir.create(data_type, showWarnings = FALSE)
  write.table(locs[ind_tmp, ], file = file.path(data_type, "X.csv"), sep = ",",
              row.names = F, col.names = F)
  write.table(y[ind_tmp], file = file.path(data_type, "y.csv"), sep = ",",
              row.names = F, col.names = F)
}
