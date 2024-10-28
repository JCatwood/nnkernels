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
  df$method[df$method == "NNSPGP"] <- "NN-SPGP"
  df$method <- factor(df$method, levels = unique(df$method))
  df[["RMSE"]] <- sqrt(df$MSE)
  
  ggplot(data = df, mapping = aes(x = m, y = RMSE)) +
    geom_line(mapping = aes(colour = method, group = method), linewidth = 1.4) +
    theme_minimal() +
    theme(
      text = element_text(size = 18),
      legend.title = element_blank(),
      legend.position = c(0.8, 0.8),
      plot.title = element_text(hjust = 0.5)
    )
  ggsave(paste0(scene, "_RMSE.pdf"), width = 5, height = 5)
}
