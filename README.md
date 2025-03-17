# Ensemble-of-GCMs
# ----------------------------------
# 1. Load Required Libraries
# ----------------------------------
# Install missing packages
packages <- c("caret", "ggplot2", "reshape2", "Metrics", "dplyr", "BMA")
installed_packages <- rownames(installed.packages())
for (p in packages) {
  if (!(p %in% installed_packages)) {
    install.packages(p, dependencies = TRUE)
  }
}
library(caret)    # Machine Learning (Super Ensemble)
library(ggplot2)  # Visualization
library(reshape2) # Data transformation for visualization
library(Metrics)  # RMSE, MAE, MBE calculations
library(dplyr)    # Data manipulation
library(BMA)      # Bayesian Model Averaging

# ----------------------------------
# 2. Load Data from CSV File
# ----------------------------------
climate_data <- read.csv("climate_data.csv", header = TRUE)

# Convert all columns to numeric (to avoid data type issues)
climate_data <- climate_data %>%
  mutate(across(everything(), as.numeric))  

# Check data structure
str(climate_data)  
head(climate_data)

# ----------------------------------
# 3. Compute Ensemble Predictions
# ----------------------------------

# --- 3.1. Weighted Averaging Ensemble ---
# Calculate RMSE for GCM1 and GCM2
rmse_values <- sapply(climate_data[, c("GCM1", "GCM2")], function(model_pred) rmse(model_pred, climate_data$Observed))

# Compute inverse-RMSE weights (normalized)
weights <- (1 / rmse_values) / sum(1 / rmse_values)
cat("Weighted Averaging Weights: GCM1 =", round(weights[1], 3), "GCM2 =", round(weights[2], 3), "\n")

# Compute weighted ensemble prediction
climate_data$Weighted_Ensemble <- climate_data$GCM1 * weights[1] + climate_data$GCM2 * weights[2]

# --- 3.2. Super Ensemble Model using Linear Regression ---
# Train a linear model combining GCM1 and GCM2
reg_model <- train(Observed ~ GCM1 + GCM2, data = climate_data, method = "lm")

# Predict using the trained model
climate_data$Super_Ensemble <- predict(reg_model, newdata = climate_data)

# --- 3.3. Bayesian Model Averaging (BMA) Ensemble ---
# Fit BMA model using bicreg (Bayesian Model Averaging)
bma_model <- bicreg(x = climate_data[, c("GCM1", "GCM2")], y = climate_data$Observed)

# Predict using BMA model (posterior mean predictions)
climate_data$BMA_Ensemble <- predict(bma_model, newdata = climate_data[, c("GCM1", "GCM2")])$mean

# ----------------------------------
# 4. Compute Error Metrics for Evaluation
# ----------------------------------

# Define function to calculate MBE
mbe <- function(pred, obs) {
  mean(pred - obs, na.rm = TRUE)  # Ensure no NA values affect calculations
}

# Define function to calculate R² (coefficient of determination)
r_squared <- function(pred, obs) {
  if (is.numeric(pred) && is.numeric(obs)) {
    cor(pred, obs, use = "complete.obs") ^ 2  # Handle missing values
  } else {
    return(NA)
  }
}

# List of models to evaluate (including BMA)
models <- c("GCM1", "GCM2", "Weighted_Ensemble", "Super_Ensemble", "BMA_Ensemble")

# Initialize empty dataframe to store results
evaluation_results <- data.frame(Model = character(),
                                 RMSE = numeric(),
                                 MAE = numeric(),
                                 MBE = numeric(),
                                 R2 = numeric(),
                                 stringsAsFactors = FALSE)

# Loop through each model and compute metrics
for (model in models) {
  if (model %in% colnames(climate_data)) {  # Ensure column exists
    rmse_val <- rmse(climate_data[[model]], climate_data$Observed)
    mae_val <- mae(climate_data[[model]], climate_data$Observed)
    mbe_val <- mbe(climate_data[[model]], climate_data$Observed)
    r2_val <- r_squared(climate_data[[model]], climate_data$Observed)

    evaluation_results <- rbind(evaluation_results, data.frame(Model = model,
                                                                RMSE = rmse_val,
                                                                MAE = mae_val,
                                                                MBE = mbe_val,
                                                                R2 = r2_val))
  }
}

# Print evaluation results
print(evaluation_results)

# Save evaluation results to CSV
write.csv(evaluation_results, "evaluation_results.csv", row.names = FALSE)

# ----------------------------------
# 5. Export Predictions to CSV (Structured by Method)
# ----------------------------------
valid_cols <- c("Year", "Month", "Observed", "GCM1", "GCM2", 
                "Weighted_Ensemble", "Super_Ensemble", "BMA_Ensemble")

valid_cols <- valid_cols[valid_cols %in% colnames(climate_data)]  # Ensure columns exist

write.csv(climate_data[, valid_cols], "ensemble_results.csv", row.names = FALSE)
cat("Ensemble results saved to ensemble_results.csv\n")

# ----------------------------------
# 6. Visualization (Comparing Methods Separately)
# ----------------------------------
# Ensure required libraries are loaded
library(ggplot2)
library(reshape2)

# Transform data for ggplot2
melted_data <- melt(climate_data, id.vars = c("Year", "Month", "Observed"),
                    measure.vars = c("GCM1", "GCM2", "Weighted_Ensemble", "Super_Ensemble", "BMA_Ensemble"),
                    variable.name = "Model", value.name = "Prediction")

# Create scatter plot of Observed vs. Predicted values (Each Method Separately)
ggplot(melted_data, aes(x = Observed, y = Prediction, color = Model)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed") +
  geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "black") +
  labs(title = "Observed vs. Predicted Values by Method",
       x = "Observed Temperature",
       y = "Predicted Temperature") +
  theme_minimal()
  
