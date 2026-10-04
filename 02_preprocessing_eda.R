# =============================================================================
# Indian Railways: Predict Train Delay
# FILE 02  --  PART 2: DATA PREPROCESSING + EDA   (Eisha)
#
# Input : cleaned_data.rds   (made by 01_understanding_cleaning.R)
# Output: plots/ folder with EDA charts, preprocessed_data.rds
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))
required <- c("tidyverse", "corrplot")
missing_pkgs <- required[!required %in% rownames(installed.packages())]
if (length(missing_pkgs)) install.packages(missing_pkgs)
library(tidyverse)
library(corrplot)
set.seed(42)
dir.create("plots", showWarnings = FALSE)

# ---- CONFIG (already filled for this dataset) -------------------------------
TRAIN_PATH  <- "ir_train.csv"
TEST_PATH   <- "ir_test.csv"
DICT_PATH   <- "ir_data_dictionary.csv"
SAMPLE_SUB  <- "ir_sample_submission.csv"

TARGET      <- "is_delayed"                  # 1 = arrived > 15 min late
ID_COL      <- "journey_id"
DATE_COLS   <- c("departure_date")
# Columns that only exist AFTER the journey (data leakage) -> must be removed
LEAKY_COLS  <- c("delay_minutes", "primary_delay_cause")
# Columns that look like numbers but are really labels (many categories)
HIGH_CARD_COLS <- c("train_number")
# Redundant columns (zone is the long name of zone_abbr)
DROP_COLS   <- c("zone")

# SAMPLE_N: the train file has 1.5 million rows. Start with 200,000 so that
# everything runs in minutes. Set to NULL later to use ALL rows.
SAMPLE_N    <- 200000

RUN_PCA          <- TRUE      # run section 5B
USE_PCA_FEATURES <- TRUE      # add PC scores as model features
PCA_VARIANCE     <- 0.95      # keep PCs that explain 95% of variance

# helper: make column names lowercase_with_underscores
clean_nm <- function(x) tolower(gsub("^_|_$", "", gsub("[^A-Za-z0-9]+", "_", x)))

# ---- Load the cleaned data produced by Part 1 -------------------------------
if (!file.exists("cleaned_data.rds"))
  stop("cleaned_data.rds not found. Run 01_understanding_cleaning.R first ",
       "(or get cleaned_data.rds from the teammate who did Part 1).")
cleaned <- readRDS("cleaned_data.rds")
train <- cleaned$train
test  <- cleaned$test
cat("Loaded cleaned data:", nrow(train), "rows x", ncol(train), "columns\n")

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
# Everything is LEARNED on train only and then re-used on test (no leakage).
feat_cols <- setdiff(names(train), c("is_late", ID_COL))
num_cols  <- feat_cols[sapply(train[feat_cols], is.numeric)]
cat_cols  <- feat_cols[sapply(train[feat_cols],
                              function(x) is.character(x) || is.factor(x) || is.logical(x))]

# 3.1 values used to fill missing data
medians <- sapply(train[num_cols], median, na.rm = TRUE)
get_mode <- function(x) names(sort(table(x), decreasing = TRUE))[1]
modes   <- sapply(train[cat_cols], get_mode)
na_cols <- num_cols[sapply(train[num_cols], anyNA)]       # which numerics had NAs

# 3.2 outlier caps (1st and 99th percentile) for continuous numeric columns
cont_pre <- num_cols[sapply(train[num_cols], function(x) n_distinct(x) > 10)]
caps <- lapply(train[cont_pre], quantile, probs = c(0.01, 0.99), na.rm = TRUE)

# 3.3 frequency maps for high-cardinality labels (e.g. train_number)
freq_maps <- lapply(train[intersect(HIGH_CARD_COLS, names(train))],
                    function(x) prop.table(table(x)))

# 3.4 ordinal encoding of station categories (A1 best ... E lowest)
station_rank <- c(A1 = 6, A = 5, B = 4, C = 3, D = 2, E = 1)

preprocess <- function(df) {
  for (col in na_cols) if (col %in% names(df))
    df[[paste0(col, "_was_na")]] <- as.integer(is.na(df[[col]]))        # NA flag
  for (col in num_cols) if (col %in% names(df))
    df[[col]][is.na(df[[col]])] <- medians[[col]]                        # impute
  for (col in names(caps)) if (col %in% names(df))
    df[[col]] <- pmin(pmax(df[[col]], caps[[col]][1]), caps[[col]][2])   # cap outliers
  for (col in cat_cols) if (col %in% names(df)) {
    df[[col]] <- as.character(df[[col]])
    df[[col]][is.na(df[[col]])] <- modes[[col]]                          # impute
  }
  if ("source_station_category" %in% names(df))
    df$source_station_rank <- unname(station_rank[df$source_station_category])
  if ("destination_station_category" %in% names(df))
    df$destination_station_rank <- unname(station_rank[df$destination_station_category])
  df
}

train <- preprocess(train)
if (!is.null(test)) test <- preprocess(test)

cat("\nMissing values left after preprocessing:", sum(is.na(train)), "\n")
cat("Numeric columns:", length(num_cols), "| Categorical columns:", length(cat_cols), "\n")

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (plots are saved in the 'plots' folder)
# =============================================================================
eda <- slice_sample(train, n = min(nrow(train), 100000))   # sample keeps plots fast

is_binary <- function(x) is.numeric(x) && all(unique(na.omit(x)) %in% c(0, 1))
bin_cols  <- names(train)[sapply(train, is_binary)]
bin_cols  <- setdiff(bin_cols, "is_late")
bin_cols  <- bin_cols[!grepl("_was_na$", bin_cols)]
cont_cols <- names(train)[sapply(train, function(x) is.numeric(x) && n_distinct(x) > 10)]
cont_cols <- setdiff(cont_cols, c("is_late", ID_COL))
cont_cols <- cont_cols[!grepl("_was_na$", cont_cols)]

# 4.1 Target distribution
p <- train %>% count(is_late) %>% mutate(pct = n / sum(n)) %>%
  ggplot(aes(factor(is_late), n, fill = factor(is_late))) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = scales::percent(pct, accuracy = 0.1)), vjust = -0.3) +
  labs(title = "Target distribution", x = "Late > 15 min (1 = yes)", y = "Journeys")
ggsave("plots/01_target.png", p, width = 5, height = 4)

# 4.2 Distributions of continuous variables
p <- eda %>% select(all_of(cont_cols)) %>% pivot_longer(everything()) %>%
  ggplot(aes(value)) + geom_histogram(bins = 40, fill = "steelblue") +
  facet_wrap(~name, scales = "free") + labs(title = "Distributions of continuous variables")
ggsave("plots/02_continuous_hist.png", p, width = 14, height = 10)

# 4.3 Continuous variables vs target
p <- eda %>% select(all_of(cont_cols), is_late) %>% pivot_longer(-is_late) %>%
  ggplot(aes(factor(is_late), value, fill = factor(is_late))) +
  geom_boxplot(outlier.alpha = 0.1, show.legend = FALSE) +
  facet_wrap(~name, scales = "free_y") + labs(title = "Continuous variables vs target", x = "is_late")
ggsave("plots/03_continuous_vs_target.png", p, width = 14, height = 10)

# 4.4 Correlation matrix (continuous + target)
png("plots/04_correlation.png", 1100, 1100)
corrplot(cor(eda[c(cont_cols, "is_late")], use = "pairwise.complete.obs"),
         method = "color", type = "upper", tl.cex = 0.8, tl.col = "black")
dev.off()

cors <- cor(eda[cont_cols], eda$is_late, use = "pairwise.complete.obs")
cat("\nCorrelation of each continuous variable with is_late:\n")
print(round(sort(cors[, 1], decreasing = TRUE), 3))

# 4.5 Late rate for binary flags
p <- eda %>% select(is_late, all_of(bin_cols)) %>% pivot_longer(-is_late) %>%
  group_by(name, value) %>% summarise(rate = mean(is_late), .groups = "drop") %>%
  ggplot(aes(factor(value), rate, fill = factor(value))) +
  geom_col(show.legend = FALSE) + facet_wrap(~name) +
  labs(title = "Late rate for each yes/no feature", x = "feature value (0/1)", y = "Share late")
ggsave("plots/05_binary_late_rate.png", p, width = 14, height = 10)

# 4.6 Late rate by category (columns with <= 30 distinct values)
for (col in cat_cols) if (n_distinct(train[[col]]) <= 30) {
  p <- train %>% group_by(.data[[col]]) %>%
    summarise(late_rate = mean(is_late), n = n(), .groups = "drop") %>%
    ggplot(aes(reorder(.data[[col]], late_rate), late_rate)) +
    geom_col(fill = "tomato") + coord_flip() +
    labs(title = paste("Late rate by", col), x = col, y = "Share late")
  ggsave(sprintf("plots/06_late_rate_by_%s.png", col), p, width = 8, height = 5)
}

# 4.7 Time patterns: hour, weekday, month, year
for (col in intersect(c("departure_hour", "day_of_week", "month", "year"), names(train))) {
  p <- train %>% group_by(.data[[col]]) %>% summarise(late_rate = mean(is_late), .groups = "drop") %>%
    ggplot(aes(.data[[col]], late_rate)) + geom_line(colour = "darkblue") + geom_point() +
    labs(title = paste("Late rate by", col), y = "Share late")
  ggsave(sprintf("plots/07_late_rate_by_%s.png", col), p, width = 7, height = 4)
}
if ("departure_date" %in% names(train)) {
  p <- train %>% mutate(ym = floor_date(departure_date, "month")) %>%
    group_by(ym) %>% summarise(late_rate = mean(is_late), .groups = "drop") %>%
    ggplot(aes(ym, late_rate)) + geom_line() + geom_point() +
    labs(title = "Monthly late rate over time", x = "month", y = "Share late")
  ggsave("plots/08_monthly_trend.png", p, width = 9, height = 4)
}

# 4.8 Two-way view: season x zone heat map of late rate
if (all(c("season", "zone_abbr") %in% names(train))) {
  p <- train %>% group_by(season, zone_abbr) %>%
    summarise(late_rate = mean(is_late), .groups = "drop") %>%
    ggplot(aes(zone_abbr, season, fill = late_rate)) + geom_tile() +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Late rate: season x zone")
  ggsave("plots/09_season_zone_heatmap.png", p, width = 9, height = 5)
}

# 4.9 Summary tables saved as CSV (handy for your report)
write_csv(as_tibble(summary(select(train, all_of(cont_cols))) %>% as.data.frame.matrix(),
                    rownames = "stat"), "plots/continuous_summary.csv")
write_csv(tibble(feature = names(cors[, 1]), corr_with_target = as.numeric(cors[, 1])) %>%
            arrange(desc(abs(corr_with_target))), "plots/correlation_with_target.csv")

cat("\nEDA finished. Open the 'plots' folder to see the charts.\n")


# =============================================================================
# SAVE RESULTS FOR TEAMMATES (feature engineering / modelling part)
# =============================================================================
saveRDS(list(train = train, test = test, num_cols = num_cols, cat_cols = cat_cols,
             medians = medians, modes = modes, caps = caps, freq_maps = freq_maps,
             na_cols = na_cols, station_rank = station_rank),
        "preprocessed_data.rds")
cat("\nSaved preprocessed_data.rds. PREPROCESSING + EDA DONE.\n")
