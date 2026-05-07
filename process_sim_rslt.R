library(jsonlite)
library(dplyr)
library(purrr)
library(tidyr)
library(knitr)
library(kableExtra)
library(ggplot2)

model_order <- c("VGP", "VGP_SM", "VGP_Wilson2015Deep", "DeepKernelNNGP", "VGP_true")
kernel_sim_include <- c("MyMaternKernel", "MyNSKernel_Lengthscale", "PeriodicKernel", "TransformedMaternKernel")
model_labels <- c(
  VGP = "MT15",
  VGP_SM = "SM",
  VGP_Wilson2015Deep = "DTSM",
  DeepKernelNNGP = "NeuVec",
  VGP_true = "True"
)
kernel_sim_labels <- c(
  MyMaternKernel = "MT15",
  MyNSKernel_Lengthscale = "RangeNS",
  PeriodicKernel = "Periodic",
  TransformedMaternKernel = "DTMT15"
)
file_path <- "results.txt"
lines <- readLines(file_path)
lines <- lines[nzchar(lines)]
json_list <- lapply(lines, function(x) {
  obj <- fromJSON(x, simplifyVector = TRUE)
  
  # Remove model_specs if it exists
  obj$model_specs <- NULL
  
  return(obj)
})
# Convert each to data frame
df_list <- lapply(json_list, function(x) {
  as.data.frame(x, stringsAsFactors = FALSE)
})
# Bind rows, filling missing columns with NA
df_all <- bind_rows(df_list) %>% mutate(model = factor(model, levels = model_order)) %>%
  mutate(model_label = model_labels[as.character(model)])

# simulation table -----------------------------
cleaned_sim_rslt = df_all %>% filter(data_type == "simulation") %>% 
  distinct(kernel_sim, model, m, .keep_all = TRUE)
## table at m = 30 -----------------------------
df = cleaned_sim_rslt %>% filter(m == 30) %>% 
  filter(kernel_sim %in% kernel_sim_include) %>% select(kernel_sim, model_label, NLL, MSE)
df_table <- df %>%
  pivot_wider(
    id_cols = kernel_sim,
    names_from = model_label,
    values_from = c(MSE, NLL),
    names_glue = "{model_label}_{.value}"
  ) %>%
  arrange(kernel_sim) %>%
  mutate(kernel_sim = kernel_sim_labels[kernel_sim])
## find the cell to make bold ----------------
comparison_models <- c("MT15", "SM", "DTSM", "NeuVec")
mse_cols <- paste0(comparison_models, "_MSE")
nll_cols <- paste0(comparison_models, "_NLL")
format_bold <- function(x, best) {
  ifelse(
    is.na(x), "",
    ifelse(
      abs(x - best) < 1e-10,
      paste0("\\textbf{", sprintf("%.3f", x), "}"),
      sprintf("%.3f", x)
    )
  )
}
df_table = df_table %>% rowwise() %>%
  mutate(
    best_MSE = min(c_across(all_of(mse_cols)), na.rm = TRUE),
    best_NLL = min(c_across(all_of(nll_cols)), na.rm = TRUE)
  ) %>%
  ungroup()
for (col in mse_cols) {
  df_table[[col]] <- format_bold(df_table[[col]], df_table$best_MSE)
}
for (col in nll_cols) {
  df_table[[col]] <- format_bold(df_table[[col]], df_table$best_NLL)
}
df_table <- df_table %>%
  select(-best_MSE, -best_NLL)
## output latex table -------------------------
df_table <- df_table[, c("kernel_sim", paste0(model_labels, "_MSE"), paste0(model_labels, "_NLL"))]
colnames(df_table) <- c(
  "Kernel",
  rep(model_labels, 2)
)
latex_table <- df_table %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    digits = 3,
    escape = FALSE,
    caption = "Prediction accuracy by simulation kernel and model."
  ) %>%
  add_header_above(c(" " = 1, "MSE" = 5, "NLL" = 5))
cat(latex_table)

# simulation figure -----------------------------
cleaned_sim_rslt = df_all %>% filter(data_type == "simulation") %>% 
  distinct(kernel_sim, model, m, .keep_all = TRUE)
comparison_models <- c("MT15", "SM", "DTSM", "NeuVec")
df = cleaned_sim_rslt %>% filter(kernel_sim %in% kernel_sim_include) %>% 
  filter(model_label %in% comparison_models) %>%
  select(kernel_sim, model_label, m, NLL, MSE)
df <- df %>%
  mutate(
    kernel_sim = kernel_sim_labels[as.character(kernel_sim)],
    kernel_sim = factor(kernel_sim, levels = kernel_sim_labels),
    model_label = factor(model_label, levels = comparison_models)
  )
m_breaks <- sort(unique(df$m))
p_mse <- ggplot(df, aes(x = m, y = MSE, color = model_label, group = model_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ kernel_sim, scales = "free_y") +
  scale_x_continuous(breaks = m_breaks) +
  labs(x = "m", y = "MSE", color = "Model") +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.box.margin = margin(t = -8),
    legend.margin = margin(t = -5),
    legend.spacing.y = unit(0.1, "cm")
  )

p_nll <- ggplot(df, aes(x = m, y = NLL, color = model_label, group = model_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ kernel_sim, scales = "free_y") +
  scale_x_continuous(breaks = m_breaks) +
  labs(x = "m", y = "NLL", color = "Model") +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.box.margin = margin(t = -8),
    legend.margin = margin(t = -5),
    legend.spacing.y = unit(0.1, "cm")
  )
if (!dir.exists("plots")) {
  dir.create("plots")
}
ggsave("plots/MSE_vs_m_by_kernel.pdf", p_mse, width = 8, height = 5)
ggsave("plots/NLL_vs_m_by_kernel.pdf", p_nll, width = 8, height = 5)

# application table -----------------------------
cleaned_sim_rslt = df_all %>% filter(data_type == "data") %>% 
  distinct(kernel_sim, model, m, .keep_all = TRUE)
comparison_models <- c("MT15", "SM", "DTSM", "NeuVec")
df = cleaned_sim_rslt %>%
  filter(model_label %in% comparison_models) %>%
  select(data_name, model_label, NLL, MSE)
df_table = df %>% pivot_wider(
  id_cols = data_name, 
  names_from = model_label,
  values_from = c(MSE, NLL),
  names_glue = "{model_label}_{.value}"
  )
## find the cell to make bold ----------------
mse_cols <- paste0(comparison_models, "_MSE")
nll_cols <- paste0(comparison_models, "_NLL")
format_bold <- function(x, best) {
  ifelse(
    is.na(x), "",
    ifelse(
      abs(x - best) < 1e-10,
      paste0("\\textbf{", sprintf("%.3f", x), "}"),
      sprintf("%.3f", x)
    )
  )
}
df_table = df_table %>% rowwise() %>%
  mutate(
    best_MSE = min(c_across(all_of(mse_cols)), na.rm = TRUE),
    best_NLL = min(c_across(all_of(nll_cols)), na.rm = TRUE)
  ) %>%
  ungroup()
for (col in mse_cols) {
  df_table[[col]] <- format_bold(df_table[[col]], df_table$best_MSE)
}
for (col in nll_cols) {
  df_table[[col]] <- format_bold(df_table[[col]], df_table$best_NLL)
}
df_table <- df_table %>%
  select(-best_MSE, -best_NLL)
## output table -------------
df_table <- df_table[, c(paste0(comparison_models, "_MSE"), paste0(comparison_models, "_NLL"))]
colnames(df_table) <- c(rep(comparison_models, 2))
latex_table <- df_table %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    digits = 3,
    escape = FALSE,
    caption = "Prediction accuracy for Argo dataset."
  ) %>%
  add_header_above(c("MSE" = 4, "NLL" = 4))
cat(latex_table)



