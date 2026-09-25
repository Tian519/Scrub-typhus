#!/usr/bin/env Rscript

# Fit 20 randomForestSRC models and collect subsampling variable importance.
# Usage: Rscript 01_fit_rf_models.R /path/to/restricted_input.csv output
# Never commit the input CSV or generated model objects: they are derived from
# restricted surveillance records.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript 01_fit_rf_models.R INPUT_CSV OUTPUT_DIRECTORY")
}
input_file <- args[[1L]]
output_dir <- args[[2L]]
if (!file.exists(input_file)) stop("Input file does not exist: ", input_file)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(randomForestSRC)
  library(caTools)
})

data <- read.csv(input_file, check.names = FALSE)
required <- c(
  "Cases", "pop2010", "deforestation", "Grassland_degradation",
  "Cultivation_reduction", "Month", "Ld", "Ls", "pre_3mon",
  "tem_3mon", "DEM", "DistanceToCoast", "Forest", "Grassland",
  "Cropland", "Urban_development"
)
missing_columns <- setdiff(required, names(data))
if (length(missing_columns)) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}
if (anyNA(data[, required]) || any(data$pop2010 <= 0) ||
    any(data$Cases < 0)) {
  stop("Required fields must be complete; population must be positive and cases nonnegative.")
}

data$pop_log <- log10(data$pop2010)
data$deforestation_abs <- abs(data$deforestation)
data$Grassland_degradation_abs <- abs(data$Grassland_degradation)
data$Cultivation_reduction_abs <- abs(data$Cultivation_reduction)
data$Month <- factor(data$Month)
data$Ld <- factor(data$Ld)
data$Ls <- factor(data$Ls)

# ST is used for stratification of the 80/20 split; Cases remains the outcome.
data$ST <- factor(ifelse(data$Cases > 0, 1, 0))
if (length(unique(data$ST)) != 2L) {
  stop("Both zero-case and positive-case rows are needed for stratified splitting.")
}

model_formula <- Cases ~
  pre_3mon + tem_3mon + DEM + DistanceToCoast + Forest + Grassland +
  pop_log + Cropland + Urban_development + deforestation_abs +
  Grassland_degradation_abs + Cultivation_reduction_abs + Ld + Ls + Month

rf_models <- vector("list", 20L)
importance_tables <- vector("list", 20L)

for (i in seq_len(20L)) {
  set.seed(i * 123L)
  in_training <- caTools::sample.split(data$ST, SplitRatio = 0.8)
  training <- data[in_training, , drop = FALSE]
  # The held-out set is reserved for evaluation in a separate script.
  rf_models[[i]] <- randomForestSRC::rfsrc(
    model_formula,
    data = training,
    ntree = 500,
    do.trace = FALSE,
    importance = "random",
    statistics = TRUE
  )

  sampled <- randomForestSRC::subsample(rf_models[[i]], verbose = FALSE)
  importance <- randomForestSRC::extract.subsample(sampled)$var.jk.sel.Z
  importance$feature <- rownames(importance)
  importance$run <- i
  importance_tables[[i]] <- importance
  message("Completed model ", i, "/20")
}

# Restricted-data-derived outputs must remain outside the public repository.
saveRDS(rf_models, file.path(output_dir, "rf_models.rds"))
write.csv(
  do.call(rbind, importance_tables),
  file.path(output_dir, "variable_importance.csv"),
  row.names = FALSE
)
