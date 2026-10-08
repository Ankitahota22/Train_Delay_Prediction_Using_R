library(ggplot2)

comparison <- read.csv("model_results/model_comparison.csv", stringsAsFactors = FALSE)

metrics <- c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC")

plot_df <- comparison[, c("Model", metrics)]
plot_df_long <- reshape(
  data = plot_df,
  varying = metrics,
  v.names = "Value",
  timevar = "Metric",
  idvar = "Model",
  times = metrics,
  direction = "long"
)

plot_df_long$Metric <- factor(plot_df_long$Metric, levels = metrics)
plot_df_long$Model <- factor(plot_df_long$Model, levels = comparison$Model)

plot_df_long$Value <- as.numeric(plot_df_long$Value)

p <- ggplot(
  plot_df_long,
  aes(x = Metric, y = Value, fill = Model)
) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, colour = "black") +
  geom_text(
    aes(label = sprintf("%.3f", Value)),
    position = position_dodge(width = 0.8),
    vjust = -0.35,
    size = 3,
    colour = "black"
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    labels = function(x) paste0(round(x * 100, 0), "%")
  ) +
  labs(
    title = "Predictive Model Comparison",
    subtitle = "Model performance by key evaluation metrics",
    x = "Metric",
    y = "Score",
    fill = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.text.x = element_text(angle = 25, hjust = 1),
    legend.position = "bottom"
  )

png(
  "model_results/plots/model_metric_comparison.png",
  width = 1100,
  height = 700,
  res = 150
)
print(p)
dev.off()

cat("Saved model comparison plot to model_results/plots/model_metric_comparison.png\n")
