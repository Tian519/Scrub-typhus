rm(list = ls())
gc()

# =========================================================
# Load required packages
# =========================================================
suppressPackageStartupMessages({
  library(randomForestSRC)
  library(fastshap)
  library(shapviz)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

# =========================================================
# Configure input and output paths
# =========================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("Usage: Rscript RF_SHAP.R DATA_CSV MODELS_FILE RESULT_DIR FIGURE_DIR")
data_path <- args[[1]]
model_path <- args[[2]]
out_dir_res <- args[[3]]
out_dir_fig <- args[[4]]

if (!dir.exists(out_dir_fig)) dir.create(out_dir_fig, recursive = TRUE)
if (!dir.exists(out_dir_res)) dir.create(out_dir_res, recursive = TRUE)

# =========================================================
# Read data and apply the same preprocessing as model fitting
# =========================================================
data <- read.csv(data_path, stringsAsFactors = FALSE)

data$pop_log <- log10(data$pop2010)
data$deforestation_abs <- abs(data$deforestation)
data$Grassland_degradation_abs <- abs(data$Grassland_degradation)
data$Cultivation_reduction_abs <- abs(data$Cultivation_reduction)

data$Month <- as.factor(data$Month)
data$Year  <- as.factor(data$Year)
data$Ld    <- as.factor(data$Ld)
data$Ls    <- as.factor(data$Ls)
data$ST    <- as.factor(ifelse(data$Cases > 0, 1, 0))

# =========================================================
# Use the model predictors for SHAP
# Keep Month in the SHAP calculation; exclude it only from plots
# =========================================================
feature_names <- c(
  "pre_3mon", "tem_3mon",
  "DEM", "DistanceToCoast",
  "Forest", "Grassland",
  "pop_log", "Cropland",
  "Urban_development",
  "deforestation_abs", "Grassland_degradation_abs",
  "Cultivation_reduction_abs",
  "Ld", "Ls", "Month"
)

# Exclude Month from visualizations
plot_features <- setdiff(feature_names, "Month")

# Select numeric predictors for dependence plots; exclude Month
numeric_features <- c(
  "pre_3mon", "tem_3mon", "DEM", "DistanceToCoast",
  "Forest", "Grassland", "pop_log", "Cropland",
  "Urban_development", "deforestation_abs",
  "Grassland_degradation_abs", "Cultivation_reduction_abs"
)

# SHAP input: retain the original predictor names
X <- data[, feature_names, drop = FALSE]

# Display labels
pretty_names <- c(
  pre_3mon = "Precipitation (3-mon)",
  tem_3mon = "Temperature (3-mon)",
  DEM = "Elevation",
  DistanceToCoast = "Distance to coast",
  Forest = "Forest",
  Grassland = "Grassland",
  pop_log = "Population (log10)",
  Cropland = "Cropland",
  Urban_development = "Urban development",
  deforestation_abs = "Deforestation",
  Grassland_degradation_abs = "Grassland degradation",
  Cultivation_reduction_abs = "Cultivation reduction",
  Ld = "Ld",
  Ls = "Ls",
  Month = "Month"
)

# =========================================================
# Load fitted models
# =========================================================
if (grepl("\\.rds$", model_path, ignore.case = TRUE)) {
  RF_list <- readRDS(model_path)
} else {
  model_env <- new.env(parent = emptyenv())
  load(model_path, envir = model_env)
  if (exists("RF_list", envir = model_env, inherits = FALSE)) {
    RF_list <- model_env$RF_list
  }
}

if (!exists("RF_list")) stop("RF_list was not found in the model file.")
if (length(RF_list) == 0) stop("RF_list is empty.")

cat("Number of models:", length(RF_list), "\n")

# =========================================================
# Compare out-of-bag errors and select a model
# =========================================================
get_oob_error <- function(mod) {
  if (!is.null(mod$error.rate)) {
    er <- as.numeric(mod$error.rate)
    er <- er[is.finite(er)]
    if (length(er) > 0) return(tail(er, 1))
  }
  return(Inf)
}

oob_errors <- sapply(RF_list, get_oob_error)

oob_df <- data.frame(
  model_id = seq_along(RF_list),
  oob_error = oob_errors
)

write.csv(
  oob_df,
  file = file.path(out_dir_res, "RF_models_OOB_error.csv"),
  row.names = FALSE
)

best_id <- which.min(oob_errors)
if (length(best_id) == 0L || !is.finite(oob_errors[best_id])) {
  stop("No model has a finite OOB error; cannot select a model.")
}
best_model <- RF_list[[best_id]]
best_oob <- oob_errors[best_id]

cat("Model with lowest OOB error:", best_id, "\n")
cat("Lowest OOB error:", best_oob, "\n")

# =========================================================
# Prediction wrapper
# Keep predictor names and factor levels consistent with model fitting
# =========================================================
pred_fun <- function(object, newdata) {
  pred <- predict(object, newdata = newdata)$predicted
  as.numeric(pred)
}

# =========================================================
# Select the same 2,000 observations for every SHAP calculation
# =========================================================
set.seed(123)
n_shap <- min(2000, nrow(X))
shap_id <- sample(seq_len(nrow(X)), n_shap)

X_shap_raw <- X[shap_id, , drop = FALSE]

# Restore the model predictor order
X_shap_raw <- X_shap_raw[, feature_names, drop = FALSE]
X <- X[, feature_names, drop = FALSE]

write.csv(
  data.frame(sample_id = shap_id),
  file = file.path(out_dir_res, "selected_2000_sample_ids.csv"),
  row.names = FALSE
)

write.csv(
  X_shap_raw,
  file = file.path(out_dir_res, "X_shap_raw_2000_samples.csv"),
  row.names = FALSE
)

# =========================================================
# SHAP settings
# =========================================================
nsim_shap <- 50   # Increase to 100 for more simulations at greater computational cost

# =========================================================
# Compute SHAP values for the model with lowest OOB error
# =========================================================
cat("Computing SHAP for the selected model...\n")

shap_best_raw <- fastshap::explain(
  object = best_model,
  X = X,
  pred_wrapper = pred_fun,
  newdata = X_shap_raw,
  nsim = nsim_shap,
  adjust = TRUE
)

saveRDS(
  shap_best_raw,
  file = file.path(out_dir_res, paste0("shap_best_model_", best_id, "_matrix.rds"))
)

shap_best_save <- as.data.frame(shap_best_raw)
shap_best_save$sample_id <- shap_id

write.csv(
  shap_best_save,
  file = file.path(out_dir_res, paste0("SHAP_values_best_model_", best_id, ".csv")),
  row.names = FALSE
)

# =========================================================
# Create a long table of feature values
# Convert mixed numeric and factor columns to character before pivoting
# =========================================================
X_shap_raw_with_id <- X_shap_raw
X_shap_raw_with_id$sample_id <- shap_id

X_long <- X_shap_raw_with_id %>%
  mutate(across(all_of(feature_names), as.character)) %>%
  pivot_longer(
    cols = all_of(feature_names),
    names_to = "feature",
    values_to = "feature_value"
  )

write.csv(
  X_long,
  file = file.path(out_dir_res, "X_long_2000_samples.csv"),
  row.names = FALSE
)

# =========================================================
# Create long-form SHAP values for the selected model
# Exclude Month from the plotting table only
# =========================================================
shap_best_long <- shap_best_save %>%
  pivot_longer(
    cols = all_of(feature_names),
    names_to = "feature",
    values_to = "shap_value"
  ) %>%
  mutate(feature_pretty = pretty_names[feature]) %>%
  left_join(X_long, by = c("sample_id", "feature")) %>%
  filter(feature %in% plot_features)

write.csv(
  shap_best_long,
  file = file.path(out_dir_res, paste0("SHAP_values_best_model_", best_id, "_long_noMonth.csv")),
  row.names = FALSE
)

# =========================================================
# Compute and save SHAP values for all 20 models
# =========================================================
cat("Computing SHAP for all 20 models...\n")

shap_array_list <- vector("list", length(RF_list))
mean_abs_shap_list <- vector("list", length(RF_list))

for (i in seq_along(RF_list)) {
  cat("Model", i, "/", length(RF_list), "\n")
  
  mod_i <- RF_list[[i]]
  
  shap_i_raw <- fastshap::explain(
    object = mod_i,
    X = X,
    pred_wrapper = pred_fun,
    newdata = X_shap_raw,
    nsim = nsim_shap,
    adjust = TRUE
  )
  
  shap_array_list[[i]] <- as.matrix(shap_i_raw)
  
  saveRDS(
    shap_i_raw,
    file = file.path(out_dir_res, paste0("shap_model_", i, "_matrix.rds"))
  )
  
  shap_i_save <- as.data.frame(shap_i_raw)
  shap_i_save$sample_id <- shap_id
  
  write.csv(
    shap_i_save,
    file = file.path(out_dir_res, paste0("SHAP_values_model_", i, ".csv")),
    row.names = FALSE
  )
  
  mean_abs_i <- colMeans(abs(shap_i_raw), na.rm = TRUE)
  
  mean_abs_shap_list[[i]] <- data.frame(
    model_id = i,
    feature = names(mean_abs_i),
    mean_abs_shap = as.numeric(mean_abs_i),
    stringsAsFactors = FALSE
  )
}

saveRDS(
  shap_array_list,
  file = file.path(out_dir_res, "all_20_models_shap_list.rds")
)

# =========================================================
# Average SHAP values across models for each observation and feature
# =========================================================
cat("Averaging SHAP matrices across 20 models...\n")

shap_dim <- dim(shap_array_list[[1]])
n_models <- length(shap_array_list)

shap_mean_matrix <- matrix(
  0,
  nrow = shap_dim[1],
  ncol = shap_dim[2],
  dimnames = list(NULL, colnames(shap_array_list[[1]]))
)

for (i in seq_along(shap_array_list)) {
  shap_mean_matrix <- shap_mean_matrix + shap_array_list[[i]]
}
shap_mean_matrix <- shap_mean_matrix / n_models

saveRDS(
  shap_mean_matrix,
  file = file.path(out_dir_res, "mean_SHAP_matrix_20_models.rds")
)

write.csv(
  data.frame(sample_id = shap_id, shap_mean_matrix),
  file = file.path(out_dir_res, "mean_SHAP_matrix_20_models.csv"),
  row.names = FALSE
)

# =========================================================
# Create the long-form averaged SHAP table
# Exclude Month from the plotting table only
# =========================================================
shap_mean_df <- as.data.frame(shap_mean_matrix)
shap_mean_df$sample_id <- shap_id

shap_mean_long <- shap_mean_df %>%
  pivot_longer(
    cols = all_of(feature_names),
    names_to = "feature",
    values_to = "shap_value"
  ) %>%
  mutate(feature_pretty = pretty_names[feature]) %>%
  left_join(X_long, by = c("sample_id", "feature")) %>%
  filter(feature %in% plot_features)

write.csv(
  shap_mean_long,
  file = file.path(out_dir_res, "mean_SHAP_long_20_models_noMonth.csv"),
  row.names = FALSE
)

# =========================================================
# Summarize mean absolute SHAP across models
# Exclude Month from the plotting summary only
# =========================================================
mean_abs_shap_all <- bind_rows(mean_abs_shap_list)

write.csv(
  mean_abs_shap_all,
  file = file.path(out_dir_res, "mean_abs_SHAP_all_20_models_long.csv"),
  row.names = FALSE
)

mean_abs_shap_summary <- mean_abs_shap_all %>%
  group_by(feature) %>%
  summarise(
    mean_abs_shap = mean(mean_abs_shap, na.rm = TRUE),
    sd_abs_shap = sd(mean_abs_shap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(feature %in% plot_features) %>%
  arrange(desc(mean_abs_shap))

mean_abs_shap_summary$feature_pretty <- pretty_names[mean_abs_shap_summary$feature]

write.csv(
  mean_abs_shap_summary,
  file = file.path(out_dir_res, "mean_abs_SHAP_summary_20_models_noMonth.csv"),
  row.names = FALSE
)

# =========================================================
# SHAP summary for the selected model (excluding Month)
# =========================================================
X_plot <- X_shap_raw[, plot_features, drop = FALSE]
colnames(X_plot) <- pretty_names[colnames(X_plot)]

shap_best_plot <- shap_best_raw[, plot_features, drop = FALSE]
colnames(shap_best_plot) <- pretty_names[colnames(shap_best_plot)]

sv_best <- shapviz(shap_best_plot, X = X_plot)

p_best_summary <- sv_importance(
  sv_best,
  kind = "beeswarm",
  max_display = ncol(X_plot)
) +
  theme_bw(base_family = "Times New Roman") +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.text = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  ) +
  labs(
    title = paste0("Best OOB Model SHAP Summary (Model ", best_id, ")"),
    x = "SHAP value (impact on model output)",
    y = NULL
  )

ggsave(
  filename = file.path(out_dir_fig, paste0("Best_OOB_Model_", best_id, "_SHAP_summary_noMonth.jpg")),
  plot = p_best_summary,
  width = 10,
  height = 7,
  dpi = 300
)

# =========================================================
# Average SHAP summary across models (excluding Month)
# =========================================================
shap_mean_plot <- shap_mean_matrix[, plot_features, drop = FALSE]
colnames(shap_mean_plot) <- pretty_names[colnames(shap_mean_plot)]

sv_mean <- shapviz(shap_mean_plot, X = X_plot)

p_mean_summary <- sv_importance(
  sv_mean,
  kind = "beeswarm",
  max_display = ncol(X_plot)
) +
  theme_bw(base_family = "Times New Roman") +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.text = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  ) +
  labs(
    title = "Mean SHAP Summary Across 20 RF Models",
    x = "Mean SHAP value (impact on model output)",
    y = NULL
  )

ggsave(
  filename = file.path(out_dir_fig, "Mean_SHAP_summary_20_models_noMonth.jpg"),
  plot = p_mean_summary,
  width = 10,
  height = 7,
  dpi = 300
)

# =========================================================
# Average absolute SHAP bar chart (excluding Month)
# =========================================================
plot_bar <- mean_abs_shap_summary
plot_bar$feature_pretty <- factor(
  plot_bar$feature_pretty,
  levels = rev(plot_bar$feature_pretty)
)

p_abs_bar <- ggplot(plot_bar, aes(x = feature_pretty, y = mean_abs_shap)) +
  geom_col(fill = "steelblue", width = 0.75) +
  geom_errorbar(
    aes(
      ymin = pmax(mean_abs_shap - sd_abs_shap, 0),
      ymax = mean_abs_shap + sd_abs_shap
    ),
    width = 0.2,
    linewidth = 0.5
  ) +
  coord_flip() +
  theme_bw(base_family = "Times New Roman") +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.text = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    plot.title = element_text(size = 13, face = "bold", hjust = 0.5)
  ) +
  labs(
    title = "Mean Absolute SHAP Across 20 RF Models",
    x = NULL,
    y = "Mean(|SHAP value|)"
  )

ggsave(
  filename = file.path(out_dir_fig, "Mean_absolute_SHAP_barplot_20_models_noMonth.jpg"),
  plot = p_abs_bar,
  width = 9,
  height = 7,
  dpi = 300
)

# =========================================================
# Dependence-plot function
# =========================================================
make_dependence_plot <- function(df_long, feature_i, title_text, out_file) {
  
  df_sub <- df_long %>%
    filter(feature == feature_i)
  
  feature_value_num <- suppressWarnings(as.numeric(as.character(df_sub$feature_value)))
  
  if (all(is.na(feature_value_num))) {
    p <- ggplot(df_sub, aes(x = as.factor(feature_value), y = shap_value)) +
      geom_jitter(width = 0.2, height = 0, alpha = 0.5, size = 1.2, color = "#2C7FB8") +
      theme_bw(base_family = "Times New Roman") +
      theme(
        panel.grid = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
        axis.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 11, color = "black"),
        plot.title = element_text(size = 12, face = "bold", hjust = 0.5)
      ) +
      labs(
        title = title_text,
        x = pretty_names[feature_i],
        y = "SHAP value"
      )
  } else {
    df_sub$feature_value_num <- feature_value_num
    
    p <- ggplot(df_sub, aes(x = feature_value_num, y = shap_value)) +
      geom_point(alpha = 0.45, size = 1.2, color = "#2C7FB8") +
      geom_smooth(method = "loess", se = TRUE, color = "#D95F02", linewidth = 0.9) +
      theme_bw(base_family = "Times New Roman") +
      theme(
        panel.grid = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
        axis.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 11, color = "black"),
        plot.title = element_text(size = 12, face = "bold", hjust = 0.5)
      ) +
      labs(
        title = title_text,
        x = pretty_names[feature_i],
        y = "SHAP value"
      )
  }
  
  ggsave(
    filename = out_file,
    plot = p,
    width = 6.5,
    height = 5,
    dpi = 300
  )
  
  return(p)
}

# =========================================================
# Choose predictors for dependence plots
# Use the eight numeric predictors with highest mean absolute SHAP
# =========================================================
top_numeric_features <- mean_abs_shap_summary %>%
  filter(feature %in% numeric_features) %>%
  arrange(desc(mean_abs_shap)) %>%
  slice(1:8) %>%
  pull(feature)

write.csv(
  data.frame(
    feature = top_numeric_features,
    feature_pretty = pretty_names[top_numeric_features]
  ),
  file = file.path(out_dir_res, "top_numeric_features_for_dependence_noMonth.csv"),
  row.names = FALSE
)

# =========================================================
# Dependence plots for the selected model
# =========================================================
dir_best_dep <- file.path(out_dir_fig, paste0("Best_model_", best_id, "_dependence_noMonth"))
if (!dir.exists(dir_best_dep)) dir.create(dir_best_dep, recursive = TRUE)

for (f in top_numeric_features) {
  make_dependence_plot(
    df_long = shap_best_long,
    feature_i = f,
    title_text = paste0("Best OOB Model (", best_id, "): ", pretty_names[f]),
    out_file = file.path(dir_best_dep, paste0("Best_model_", best_id, "_dependence_", f, "_noMonth.jpg"))
  )
}

# =========================================================
# Dependence plots averaged across the 20 models
# =========================================================
dir_mean_dep <- file.path(out_dir_fig, "Mean_SHAP_dependence_20_models_noMonth")
if (!dir.exists(dir_mean_dep)) dir.create(dir_mean_dep, recursive = TRUE)

for (f in top_numeric_features) {
  make_dependence_plot(
    df_long = shap_mean_long,
    feature_i = f,
    title_text = paste0("Mean SHAP Across 20 Models: ", pretty_names[f]),
    out_file = file.path(dir_mean_dep, paste0("Mean_SHAP_dependence_20_models_", f, "_noMonth.jpg"))
  )
}

# =========================================================
# Save the objects needed to redraw figures
# =========================================================
plot_ready_obj <- list(
  best_model_id = best_id,
  best_oob = best_oob,
  shap_id = shap_id,
  feature_names = feature_names,
  plot_features = plot_features,
  pretty_names = pretty_names,
  numeric_features = numeric_features,
  top_numeric_features = top_numeric_features,
  X_shap_raw = X_shap_raw,
  X_long = X_long,
  shap_best_raw = shap_best_raw,
  shap_best_long = shap_best_long,
  shap_mean_matrix = shap_mean_matrix,
  shap_mean_long = shap_mean_long,
  mean_abs_shap_summary = mean_abs_shap_summary,
  oob_df = oob_df
)

saveRDS(
  plot_ready_obj,
  file = file.path(out_dir_res, "plot_ready_SHAP_objects_noMonth.rds")
)

# =========================================================
# Report completion
# =========================================================
cat("\n=============================\n")
cat("Completed.\n")
cat("Selected model ID:", best_id, "\n")
cat("Lowest OOB error:", best_oob, "\n")
cat("Sample indices have been saved.\n")
cat("Month was excluded from plots only.\n")
cat("Selected-model and average SHAP plots were saved.\n")
cat("For restyled plots, read:\n")
cat(file.path(out_dir_res, "plot_ready_SHAP_objects_noMonth.rds"), "\n")
cat("=============================\n")
