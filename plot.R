library(ggplot2)
library(RColorBrewer)

for (scene in c("Constant", "matern15", "PiecewisePolynomialKernel", 
                "RBFKernel", "exponential", "AdditiveKernel")) {
  df <- matrix(nrow = 0, ncol = 3)
  for (mtd in c("VGP", "SPGP", "NNSPGP")) {
    data <- as.matrix(read.csv(paste0(mtd, "_", scene, ".csv"), header = FALSE))
    df <- rbind(df, cbind(data, mtd))
  }
  df <- as.data.frame(df)
  colnames(df) <- c("m", "MSE", "method")
  df$m <- as.integer(df$m)
  df$MSE <- as.numeric(df$MSE)
  df$method <- factor(df$method, levels = unique(df$method))
  
  ggplot(data = df, mapping = aes(x = m, y = MSE)) +
    geom_line(mapping = aes(colour = method, group = method))
  ggsave(paste0(scene, "_MSE.pdf"), width = 5, height = 5)
}
