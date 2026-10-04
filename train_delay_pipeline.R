# ============================================
# TRAIN DELAY PREDICTION - CONFIGURATION
# ============================================

TRAIN_PATH <- "ir_train.csv"
TEST_PATH  <- "ir_test.csv"

TARGET_RAW <- "is_delayed"

ID_COL <- "journey_id"

DATE_COLS <- c("departure_date")

LEAKY_COLS <- c(
  "delay_minutes",
  "primary_delay_cause"
)
# ============================================
# LOAD REQUIRED PACKAGES
# ============================================

library(readr)
library(dplyr)
library(lubridate)
library(caret)
library(pROC)
# ============================================
# LOAD DATA
# ============================================

train <- read_csv(TRAIN_PATH)
test  <- read_csv(TEST_PATH)

cat("Train rows:", nrow(train), "\n")
cat("Train columns:", ncol(train), "\n")

cat("Test rows:", nrow(test), "\n")
cat("Test columns:", ncol(test), "\n")

print(names(train))
# ============================================
# CHECK TARGET
# ============================================

cat("\nTarget column:", TARGET_RAW, "\n")

print(table(train[[TARGET_RAW]], useNA = "ifany"))

cat("\nTarget proportion:\n")
print(prop.table(table(train[[TARGET_RAW]])))
# ============================================
# REMOVE ID AND LEAKY COLUMNS
# ============================================

remove_cols <- c(
  ID_COL,
  LEAKY_COLS
)

train_model <- train %>%
  select(-any_of(remove_cols))

test_model <- test %>%
  select(-any_of(remove_cols))

cat("\nColumns after removing ID/leaky columns:",
    ncol(train_model), "\n")

print(names(train_model))
# ============================================
# CREATE DATE FEATURES
# ============================================

train_model <- train_model %>%
  mutate(
    departure_year = year(departure_date),
    departure_month = month(departure_date),
    departure_day = day(departure_date),
    departure_week = week(departure_date)
  ) %>%
  select(-all_of(DATE_COLS))

test_model <- test_model %>%
  mutate(
    departure_year = year(departure_date),
    departure_month = month(departure_date),
    departure_day = day(departure_date),
    departure_week = week(departure_date)
  ) %>%
  select(-all_of(DATE_COLS))
# ============================================
# SEPARATE FEATURES AND TARGET
# ============================================

y <- train_model[[TARGET_RAW]]

X <- train_model %>%
  select(-all_of(TARGET_RAW))

X_test <- test_model

# ============================================
# FEATURE ENGINEERING
# ============================================

engineer_features <- function(df) {
  df %>%
    mutate(
      source_station_rank = case_when(
        source_station_category == "A1" ~ 6,
        source_station_category == "A" ~ 5,
        source_station_category == "B" ~ 4,
        source_station_category == "C" ~ 3,
        source_station_category == "D" ~ 2,
        source_station_category == "E" ~ 1,
        TRUE ~ 0
      ),
      destination_station_rank = case_when(
        destination_station_category == "A1" ~ 6,
        destination_station_category == "A" ~ 5,
        destination_station_category == "B" ~ 4,
        destination_station_category == "C" ~ 3,
        destination_station_category == "D" ~ 2,
        destination_station_category == "E" ~ 1,
        TRUE ~ 0
      ),
      station_category_gap = abs(source_station_rank - destination_station_rank),
      distance_per_hour = distance_km / pmax(scheduled_travel_hours, 1),
      distance_per_stop = distance_km / pmax(num_scheduled_stops, 1),
      crowding_pressure = seat_utilisation_pct * as.numeric(is_overloaded),
      maintenance_risk = maintenance_score * (1 + as.numeric(is_rake_shared)),
      route_risk = route_historical_ontime_pct * (1 + zone_congestion_index),
      total_weather_risk = fog_risk_score + zone_fog_index + season_severity_score,
      departure_hour_sin = sin(2 * pi * departure_hour / 24),
      departure_hour_cos = cos(2 * pi * departure_hour / 24),
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12),
      weekend_peak = as.integer(is_weekend == 1 & is_peak_hour == 1),
      night_peak = as.integer(is_night_departure == 1 & is_peak_hour == 1),
      train_age = pmax(loco_age_years, coach_age_years)
    )
}

X <- engineer_features(X)
X_test <- engineer_features(X_test)

cat("\nEngineered features added to training and test sets.\n")
print(names(X)[(length(names(X)) - 15):length(names(X))])

cat("\nNumber of training features:", ncol(X), "\n")
cat("Number of test features:", ncol(X_test), "\n")
# ============================================
# CHECK DATA TYPES
# ============================================

cat("\nData types:\n")
print(sapply(X, class))

cat("\nMissing values:\n")
print(sort(colSums(is.na(X)), decreasing = TRUE))
# ============================================
# CONVERT CHARACTER COLUMNS TO FACTORS
# ============================================

char_cols <- names(X)[sapply(X, is.character)]

for (col in char_cols) {
  X[[col]] <- as.factor(X[[col]])
  X_test[[col]] <- as.factor(X_test[[col]])
}

cat("\nCategorical columns converted:", length(char_cols), "\n")
print(char_cols)
# ============================================
# HANDLE MISSING VALUES
# ============================================

for (col in names(X)) {
  if (is.numeric(X[[col]])) {
    median_value <- median(X[[col]], na.rm = TRUE)
    X[[col]][is.na(X[[col]])] <- median_value
    X_test[[col]][is.na(X_test[[col]])] <- median_value
  }
}

cat("\nMissing values after handling:\n")
print(sum(is.na(X)))
print(sum(is.na(X_test)))
# ============================================
# CHECK TRAIN / TEST COLUMNS
# ============================================

missing_in_test <- setdiff(names(X), names(X_test))
extra_in_test <- setdiff(names(X_test), names(X))

cat("\nColumns missing in test:\n")
print(missing_in_test)

cat("\nExtra columns in test:\n")
print(extra_in_test)
# ============================================
# FEATURE EXTRACTION: PCA
# ============================================
# PCA is applied only to continuous numeric variables to reduce multicollinearity
# and capture dominant variation patterns before modeling.

numeric_cols <- names(X)[sapply(X, is.numeric)]
keep_numeric <- sapply(X[numeric_cols], function(x) length(unique(stats::na.omit(x))) > 1)
numeric_cols <- numeric_cols[keep_numeric]
cat_cols <- setdiff(names(X), numeric_cols)

if (length(numeric_cols) > 1) {
  pca_fit <- prcomp(X[numeric_cols], center = TRUE, scale. = TRUE)
  pca_summary <- summary(pca_fit)$importance
  variance_cutoff <- 0.95
  n_pca <- which(pca_summary[3, ] >= variance_cutoff)[1]
  if (is.na(n_pca)) n_pca <- ncol(pca_fit$x)

  pca_train <- as.data.frame(pca_fit$x[, 1:n_pca, drop = FALSE])
  names(pca_train) <- paste0("PC", seq_len(ncol(pca_train)))

  pca_test <- as.data.frame(predict(pca_fit, newdata = X_test[numeric_cols]))[, 1:n_pca, drop = FALSE]
  names(pca_test) <- paste0("PC", seq_len(ncol(pca_test)))

  X <- cbind(pca_train, X[cat_cols])
  X_test <- cbind(pca_test, X_test[cat_cols])

  cat("\nPCA reduced numeric features from", length(numeric_cols), "to", n_pca, "components.\n")
  cat("Cumulative variance explained:", round(pca_summary[3, n_pca], 4), "\n")
} else {
  cat("\nPCA skipped: no valid numeric features with variance were found.\n")
}
# ============================================
# CREATE TRAIN / VALIDATION SPLIT
# ============================================

set.seed(123)

train_index <- createDataPartition(
  y,
  p = 0.8,
  list = FALSE
)

X_train <- X[train_index, ]
X_valid <- X[-train_index, ]

y_train <- y[train_index]
y_valid <- y[-train_index]

cat("\nTraining rows:", nrow(X_train), "\n")
cat("Validation rows:", nrow(X_valid), "\n")
# ============================================
# TRAIN RANDOM FOREST MODEL
# ============================================

library(ranger)

set.seed(123)

sample_size <- min(200000, nrow(X_train))

sample_index <- sample(
  seq_len(nrow(X_train)),
  sample_size
)

X_train_sample <- X_train[sample_index, ]
y_train_sample <- y_train[sample_index]

model <- ranger(
  dependent.variable.name = TARGET_RAW,
  data = data.frame(
    X_train_sample,
    is_delayed = as.factor(y_train_sample)
  ),
  num.trees = 200,
  probability = TRUE,
  importance = "impurity"
)

print(model)