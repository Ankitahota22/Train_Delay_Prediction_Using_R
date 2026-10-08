
required_packages <- c(
  "readr",
  "dplyr",
  "ggplot2",
  "caret",
  "ranger",
  "pROC"
)

missing_packages <- required_packages[
  !required_packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(ranger)
library(pROC)

set.seed(123)

# Create output folders
dir.create("model_results", showWarnings = FALSE)
dir.create("model_results/plots", showWarnings = FALSE)

cat("============================================================\n")
cat("PREDICTIVE MODELLING AND VERIFICATION\n")
cat("============================================================\n")


# -----------------------------------------------------------------------------
# 2. CHECK THAT PREVIOUS PIPELINE HAS BEEN RUN
# -----------------------------------------------------------------------------

required_objects <- c("X", "X_test", "y")

missing_objects <- required_objects[
  !sapply(required_objects, exists, envir = .GlobalEnv)
]

if (length(missing_objects) > 0) {

  stop(
    paste0(
      "\nERROR: The following objects are missing: ",
      paste(missing_objects, collapse = ", "),
      "\n\nFirst run:\n",
      "source('train_delay_pipeline.R')\n"
    )
  )
}

cat("\nPrevious pipeline objects found successfully.\n")


# -----------------------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# -----------------------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("DATA INFORMATION\n")
cat("------------------------------------------------------------\n")

cat("Training rows :", nrow(X), "\n")
cat("Training cols :", ncol(X), "\n")
cat("Test rows     :", nrow(X_test), "\n")
cat("Test cols     :", ncol(X_test), "\n")

cat("\nTarget distribution:\n")
print(table(y))

cat("\nTarget proportions:\n")
print(round(prop.table(table(y)), 4))


# -----------------------------------------------------------------------------
# 4. CONVERT TARGET TO FACTOR
# -----------------------------------------------------------------------------

# We use:
# 0 = Not delayed
# 1 = Delayed

y_factor <- factor(
  y,
  levels = c(0, 1),
  labels = c("Not_Delayed", "Delayed")
)

cat("\nTarget levels:\n")
print(levels(y_factor))


# -----------------------------------------------------------------------------
# 5. CREATE TRAIN / VALIDATION SPLIT
# -----------------------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("TRAIN / VALIDATION SPLIT\n")
cat("------------------------------------------------------------\n")

set.seed(123)

train_index <- createDataPartition(
  y_factor,
  p = 0.80,
  list = FALSE
)

X_train <- X[train_index, , drop = FALSE]
X_valid <- X[-train_index, , drop = FALSE]

y_train <- y_factor[train_index]
y_valid <- y_factor[-train_index]

cat("Training rows  :", nrow(X_train), "\n")
cat("Validation rows:", nrow(X_valid), "\n")


# -----------------------------------------------------------------------------
# 6. MAKE FACTOR LEVELS CONSISTENT
# -----------------------------------------------------------------------------

# Ranger / caret require categorical variables to have compatible levels.

categorical_cols <- names(X_train)[
  sapply(X_train, function(x) is.factor(x) || is.character(x))
]

for (col in categorical_cols) {

  X_train[[col]] <- as.factor(X_train[[col]])
  X_valid[[col]] <- as.character(X_valid[[col]])
  X_valid[[col]] <- factor(
    X_valid[[col]],
    levels = levels(X_train[[col]])
  )

  X_test[[col]] <- as.character(X_test[[col]])
  X_test[[col]] <- factor(
    X_test[[col]],
    levels = levels(X_train[[col]])
  )
}


# -----------------------------------------------------------------------------
# 7. REMOVE ZERO-VARIANCE FEATURES
# -----------------------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("FEATURE CHECK\n")
cat("------------------------------------------------------------\n")

sample_n <- min(100000, nrow(X_train))

set.seed(123)
check_rows <- sample(seq_len(nrow(X_train)), sample_n)

nzv <- nearZeroVar(X_train[check_rows, , drop = FALSE])

if (length(nzv) > 0) {

  cat("Removing", length(nzv), "near-zero variance features.\n")

  X_train <- X_train[, -nzv, drop = FALSE]
  X_valid <- X_valid[, -nzv, drop = FALSE]
  X_test  <- X_test[, -nzv, drop = FALSE]

} else {

  cat("No near-zero variance features found.\n")
}

cat("Final modelling features:", ncol(X_train), "\n")


# =============================================================================
# MODEL 1 - LOGISTIC REGRESSION
# =============================================================================

# =============================================================================
# MODEL 1 - LOGISTIC REGRESSION
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 1: LOGISTIC REGRESSION\n")
cat("============================================================\n")


# -------------------------------------------------------------------------
# 1. Create a small balanced training sample
# -------------------------------------------------------------------------

set.seed(123)

# Check target classes
cat("\nTarget classes in y_train:\n")
print(table(y_train))


# Find the two actual classes
class_names <- levels(factor(y_train))

cat("\nClass names:\n")
print(class_names)


# Separate the two classes using their actual names
class_0 <- X_train[
  y_train == class_names[1],
  ,
  drop = FALSE
]

class_1 <- X_train[
  y_train == class_names[2],
  ,
  drop = FALSE
]


# Take at most 20,000 from each class
n_each <- min(
  20000,
  nrow(class_0),
  nrow(class_1)
)


cat(
  "\nRows sampled from each class:",
  n_each,
  "\n"
)


# Random samples
idx_0 <- sample(
  seq_len(nrow(class_0)),
  n_each
)

idx_1 <- sample(
  seq_len(nrow(class_1)),
  n_each
)


# Combine
X_log <- rbind(
  class_0[idx_0, , drop = FALSE],
  class_1[idx_1, , drop = FALSE]
)


# Target for logistic regression
y_log <- factor(
  c(
    rep(class_names[1], n_each),
    rep(class_names[2], n_each)
  ),
  levels = class_names
)


# Clean temporary objects
rm(
  class_0,
  class_1,
  idx_0,
  idx_1
)

gc()


cat(
  "Logistic regression training rows:",
  nrow(X_log),
  "\n"
)

# -------------------------------------------------------------------------
# 2. Create dummy variables USING ONLY PREDICTORS
# -------------------------------------------------------------------------

dummy_model <- dummyVars(
  ~ .,
  data = X_log,
  fullRank = TRUE
)


X_log_dummy <- as.data.frame(
  predict(
    dummy_model,
    newdata = X_log
  )
)


# Create validation dummy variables
X_valid_dummy <- as.data.frame(
  predict(
    dummy_model,
    newdata = X_valid
  )
)


# -------------------------------------------------------------------------
# 3. Make validation columns exactly match training columns
# -------------------------------------------------------------------------

train_cols <- colnames(X_log_dummy)

missing_cols <- setdiff(
  train_cols,
  colnames(X_valid_dummy)
)

if (length(missing_cols) > 0) {

  for (col in missing_cols) {
    X_valid_dummy[[col]] <- 0
  }

}


# Remove any extra columns
extra_cols <- setdiff(
  colnames(X_valid_dummy),
  train_cols
)

if (length(extra_cols) > 0) {

  X_valid_dummy <- X_valid_dummy[
    ,
    !(colnames(X_valid_dummy) %in% extra_cols),
    drop = FALSE
  ]

}


# Put columns in exactly the same order
X_valid_dummy <- X_valid_dummy[
  ,
  train_cols,
  drop = FALSE
]


# -------------------------------------------------------------------------
# 4. Remove near-zero variance features
# -------------------------------------------------------------------------

dummy_nzv <- nearZeroVar(
  X_log_dummy
)

if (length(dummy_nzv) > 0) {

  X_log_dummy <- X_log_dummy[
    ,
    -dummy_nzv,
    drop = FALSE
  ]

  X_valid_dummy <- X_valid_dummy[
    ,
    -dummy_nzv,
    drop = FALSE
  ]

}


cat(
  "Dummy-variable features:",
  ncol(X_log_dummy),
  "\n"
)


# -------------------------------------------------------------------------
# 5. Train Logistic Regression
# -------------------------------------------------------------------------

cat(
  "\nTraining Logistic Regression...\n"
)


logistic_train <- data.frame(
  y_log = y_log,
  X_log_dummy,
  check.names = FALSE
)


logistic_model <- glm(
  y_log ~ .,
  data = logistic_train,
  family = binomial()
)


cat(
  "\nLogistic regression trained successfully.\n"
)


# -------------------------------------------------------------------------
# 6. Predict validation data
# -------------------------------------------------------------------------

logistic_prob <- predict(
  logistic_model,
  newdata = X_valid_dummy,
  type = "response"
)


# -------------------------------------------------------------------------
# 7. Convert probabilities to classes
# -------------------------------------------------------------------------

logistic_pred <- ifelse(
  logistic_prob >= 0.50,
  "Delayed",
  "Not_Delayed"
)


logistic_pred <- factor(
  logistic_pred,
  levels = c("Not_Delayed", "Delayed")
)


# -------------------------------------------------------------------------
# 8. Confusion Matrix
# -------------------------------------------------------------------------

logistic_cm <- confusionMatrix(
  logistic_pred,
  y_valid,
  positive = "Delayed"
)


cat(
  "\nLOGISTIC REGRESSION CONFUSION MATRIX:\n"
)

print(logistic_cm$table)


# -------------------------------------------------------------------------
# 9. Metrics
# -------------------------------------------------------------------------

cat(
  "\nLOGISTIC REGRESSION METRICS:\n"
)

print(logistic_cm$byClass)


# -------------------------------------------------------------------------
# 10. ROC-AUC
# -------------------------------------------------------------------------

logistic_roc <- roc(
  response = y_valid,
  predictor = logistic_prob,
  levels = c("Not_Delayed", "Delayed"),
  direction = "<"
)


logistic_auc <- as.numeric(
  auc(logistic_roc)
)


cat(
  "\nLogistic Regression ROC-AUC:",
  round(logistic_auc, 4),
  "\n"
)


# -------------------------------------------------------------------------
# 11. Clean temporary objects
# -------------------------------------------------------------------------

rm(
  X_log_dummy,
  X_valid_dummy,
  logistic_train
)

gc()
# =============================================================================
# MODEL 2 - RANDOM FOREST
# =============================================================================

cat("\n============================================================\n")
cat("MODEL 2: RANDOM FOREST\n")
cat("============================================================\n")

set.seed(123)

# Limit training sample if dataset is extremely large.
# This follows the approach already used in your existing pipeline.

sample_size <- min(
  200000,
  nrow(X_train)
)

sample_index <- sample(
  seq_len(nrow(X_train)),
  sample_size
)

X_train_rf <- X_train[sample_index, , drop = FALSE]
y_train_rf <- y_train[sample_index]

cat("Rows used for Random Forest:", nrow(X_train_rf), "\n")


# Train Random Forest

rf_model <- ranger(
  dependent.variable.name = "target",
  data = data.frame(
    target = y_train_rf,
    X_train_rf
  ),
  num.trees = 300,
  mtry = max(
    1,
    floor(sqrt(ncol(X_train_rf)))
  ),
  min.node.size = 10,
  probability = TRUE,
  importance = "impurity",
  seed = 123
)

cat("\nRandom Forest trained successfully.\n")
print(rf_model)


# -----------------------------------------------------------------------------
# RANDOM FOREST PREDICTIONS
# -----------------------------------------------------------------------------

rf_probability <- predict(
  rf_model,
  data = X_valid
)$predictions[, "Delayed"]


rf_prediction <- ifelse(
  rf_probability >= 0.50,
  "Delayed",
  "Not_Delayed"
)

rf_prediction <- factor(
  rf_prediction,
  levels = levels(y_factor)
)


# -----------------------------------------------------------------------------
# RANDOM FOREST VERIFICATION
# -----------------------------------------------------------------------------

rf_cm <- confusionMatrix(
  rf_prediction,
  y_valid,
  positive = "Delayed"
)

cat("\nRANDOM FOREST CONFUSION MATRIX:\n")
print(rf_cm$table)

cat("\nRANDOM FOREST METRICS:\n")
print(rf_cm$byClass)

rf_roc <- roc(
  response = y_valid,
  predictor = rf_probability,
  levels = c("Not_Delayed", "Delayed"),
  direction = "<"
)

rf_auc <- as.numeric(auc(rf_roc))

cat("\nRandom Forest ROC-AUC:",
    round(rf_auc, 4), "\n")


# =============================================================================
# 8. MODEL COMPARISON
# =============================================================================

cat("\n============================================================\n")
cat("MODEL COMPARISON\n")
cat("============================================================\n")

comparison <- data.frame(

  Model = c(
    "Logistic Regression",
    "Random Forest"
  ),

  Accuracy = c(
    as.numeric(logistic_cm$overall["Accuracy"]),
    as.numeric(rf_cm$overall["Accuracy"])
  ),

  Sensitivity = c(
    as.numeric(logistic_cm$byClass["Sensitivity"]),
    as.numeric(rf_cm$byClass["Sensitivity"])
  ),

  Specificity = c(
    as.numeric(logistic_cm$byClass["Specificity"]),
    as.numeric(rf_cm$byClass["Specificity"])
  ),

  Precision = c(
    as.numeric(logistic_cm$byClass["Pos Pred Value"]),
    as.numeric(rf_cm$byClass["Pos Pred Value"])
  ),

  Recall = c(
    as.numeric(logistic_cm$byClass["Sensitivity"]),
    as.numeric(rf_cm$byClass["Sensitivity"])
  ),

  F1 = c(
    as.numeric(logistic_cm$byClass["F1"]),
    as.numeric(rf_cm$byClass["F1"])
  ),

  ROC_AUC = c(
    logistic_auc,
    rf_auc
  )
)

comparison <- comparison %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 4)
    )
  )

print(comparison)

write.csv(
  comparison,
  "model_results/model_comparison.csv",
  row.names = FALSE
)


# =============================================================================
# 9. ROC CURVE COMPARISON
# =============================================================================

cat("\nCreating ROC curve...\n")

png(
  "model_results/plots/roc_comparison.png",
  width = 900,
  height = 700
)

plot(
  logistic_roc,
  main = "ROC Curve Comparison",
  lwd = 2
)

plot(
  rf_roc,
  add = TRUE,
  lwd = 2
)

legend(
  "bottomright",
  legend = c(
    paste0("Logistic Regression AUC = ",
           round(logistic_auc, 3)),

    paste0("Random Forest AUC = ",
           round(rf_auc, 3))
  ),
  lwd = 2
)

dev.off()


# =============================================================================
# 10. CONFUSION MATRIX PLOTS
# =============================================================================

png(
  "model_results/plots/random_forest_confusion_matrix.png",
  width = 800,
  height = 700
)

fourfoldplot(
  rf_cm$table,
  color = c("grey80", "grey50"),
  main = "Random Forest Confusion Matrix"
)

dev.off()


# =============================================================================
# 11. RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================

cat("\n============================================================\n")
cat("FEATURE IMPORTANCE\n")
cat("============================================================\n")

importance_values <- ranger::importance(rf_model)

importance_df <- data.frame(
  Feature = names(importance_values),
  Importance = as.numeric(importance_values)
) %>%
  arrange(desc(Importance))

cat("\nTop 20 most important features:\n")
print(head(importance_df, 20))

write.csv(
  importance_df,
  "model_results/random_forest_feature_importance.csv",
  row.names = FALSE
)


# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# -----------------------------------------------------------------------------

top_features <- head(
  importance_df,
  20
) %>%
  arrange(Importance)

p_importance <- ggplot(
  top_features,
  aes(
    x = Importance,
    y = reorder(Feature, Importance)
  )
) +
  geom_col() +
  labs(
    title = "Top 20 Random Forest Feature Importances",
    x = "Importance",
    y = "Feature"
  ) +
  theme_minimal()

ggsave(
  "model_results/plots/feature_importance.png",
  p_importance,
  width = 10,
  height = 8
)


# =============================================================================
# 12. DELAY PROBABILITY DISTRIBUTION
# =============================================================================

prob_df <- data.frame(
  Actual = y_valid,
  Predicted_Probability = rf_probability
)

p_probability <- ggplot(
  prob_df,
  aes(
    x = Predicted_Probability,
    fill = Actual
  )
) +
  geom_histogram(
    bins = 30,
    alpha = 0.6,
    position = "identity"
  ) +
  labs(
    title = "Random Forest Predicted Delay Probability",
    x = "Probability of Delay",
    y = "Number of Trains"
  ) +
  theme_minimal()

ggsave(
  "model_results/plots/predicted_probability.png",
  p_probability,
  width = 9,
  height = 6
)


# =============================================================================
# 13. THRESHOLD ANALYSIS
# =============================================================================

cat("\n============================================================\n")
cat("THRESHOLD ANALYSIS\n")
cat("============================================================\n")

thresholds <- seq(
  0.20,
  0.80,
  by = 0.05
)

threshold_results <- data.frame()

for (threshold in thresholds) {

  pred <- ifelse(
    rf_probability >= threshold,
    "Delayed",
    "Not_Delayed"
  )

  pred <- factor(
    pred,
    levels = levels(y_factor)
  )

  cm <- confusionMatrix(
    pred,
    y_valid,
    positive = "Delayed"
  )

  threshold_results <- rbind(
    threshold_results,
    data.frame(
      Threshold = threshold,
      Accuracy = as.numeric(
        cm$overall["Accuracy"]
      ),
      Precision = as.numeric(
        cm$byClass["Pos Pred Value"]
      ),
      Recall = as.numeric(
        cm$byClass["Sensitivity"]
      ),
      F1 = as.numeric(
        cm$byClass["F1"]
      )
    )
  )
}

threshold_results <- threshold_results %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 4)
    )
  )

print(threshold_results)

write.csv(
  threshold_results,
  "model_results/threshold_analysis.csv",
  row.names = FALSE
)


# Find best threshold based on F1 score

best_threshold_row <- threshold_results[
  which.max(threshold_results$F1),
]

cat("\nBest threshold according to F1 score:\n")
print(best_threshold_row)


# =============================================================================
# 14. FINAL RANDOM FOREST MODEL
# =============================================================================

cat("\n============================================================\n")
cat("FINAL MODEL\n")
cat("============================================================\n")

final_threshold <- best_threshold_row$Threshold

cat(
  "Selected probability threshold:",
  final_threshold,
  "\n"
)


# =============================================================================
# 15. PREDICT TEST DATA
# =============================================================================

cat("\n============================================================\n")
cat("TEST DATA PREDICTION\n")
cat("============================================================\n")

test_probability <- predict(
  rf_model,
  data = X_test
)$predictions[, "Delayed"]


test_prediction <- ifelse(
  test_probability >= final_threshold,
  1,
  0
)


# -----------------------------------------------------------------------------
# SAVE PREDICTIONS
# -----------------------------------------------------------------------------

test_predictions <- data.frame(
  Predicted_Delay_Probability = test_probability,
  Predicted_Delay = test_prediction
)

write.csv(
  test_predictions,
  "model_results/test_predictions.csv",
  row.names = FALSE
)

cat(
  "\nTest predictions saved to:",
  "model_results/test_predictions.csv\n"
)


# =============================================================================
# 16. FINAL MODEL SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("FINAL MODEL SUMMARY\n")
cat("============================================================\n")

cat("\nLogistic Regression:\n")
cat("Accuracy :", round(logistic_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(logistic_cm$byClass["Pos Pred Value"], 4), "\n")
cat("Recall   :", round(logistic_cm$byClass["Sensitivity"], 4), "\n")
cat("F1       :", round(logistic_cm$byClass["F1"], 4), "\n")
cat("ROC-AUC  :", round(logistic_auc, 4), "\n")

cat("\nRandom Forest:\n")
cat("Accuracy :", round(rf_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(rf_cm$byClass["Pos Pred Value"], 4), "\n")
cat("Recall   :", round(rf_cm$byClass["Sensitivity"], 4), "\n")
cat("F1       :", round(rf_cm$byClass["F1"], 4), "\n")
cat("ROC-AUC  :", round(rf_auc, 4), "\n")

cat("\nSelected RF threshold:", final_threshold, "\n")

cat("\nTop 10 predictive features:\n")
print(head(importance_df, 10))


# =============================================================================
# 17. SAVE MODELS
# =============================================================================

saveRDS(
  logistic_model,
  "model_results/logistic_model.rds"
)

saveRDS(
  rf_model,
  "model_results/random_forest_model.rds"
)

cat("\nModels saved successfully.\n")


# =============================================================================
# DONE
# =============================================================================

cat("\n============================================================\n")
cat("PREDICTIVE MODELLING COMPLETE\n")
cat("============================================================\n")

cat("\nGenerated files:\n")
cat("1. model_results/model_comparison.csv\n")
cat("2. model_results/random_forest_feature_importance.csv\n")
cat("3. model_results/threshold_analysis.csv\n")
cat("4. model_results/test_predictions.csv\n")
cat("5. model_results/plots/roc_comparison.png\n")
cat("6. model_results/plots/random_forest_confusion_matrix.png\n")
cat("7. model_results/plots/feature_importance.png\n")
cat("8. model_results/plots/predicted_probability.png\n")
cat("\nYou can now use these results for the modelling, verification,\n")
cat("key discoveries and final verdict sections of the project.\n")