#/usr/bin/Rscript
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
setwd(direc)

palette1 <- c("firebrick3", "gray27", "forestgreen")
lines1 <- c("solid", "solid", "dotdash", "dotdash")

scatter_test <- read.csv("test_scatter.csv")

scatter_df_pred <- data.frame(
  scatter_test$predv, 
  scatter_test$predlag, 
  rep("predicted", length(scatter_test$predv)))

colnames(scatter_df_pred) <- c("Velocity", "TimeLag", "Type")

scatter_df_actual <- data.frame(
  scatter_test$actualv, 
  scatter_test$actuallag, 
  rep("actual", length(scatter_test$actualv)))

colnames(scatter_df_actual) <- c("Velocity", "TimeLag", "Type")

scatter_df <- rbind(scatter_df_pred, scatter_df_actual)

colnames(scatter_df) <- c("Velocity", "TimeLag", "Type")

ggplot(scatter_df, aes(x = Velocity, y = TimeLag, color = Type)) +
  geom_point() + 
  geom_smooth(
    method = loess, 
    method.args = list(family = "symmetric")
  ) +
  theme_gray(base_size = 23) + 
  scale_colour_manual(values=palette1) + 
  labs(y="Time Lag", x="Output") + 
  theme(legend.position="top")

ggsave("scatter_v_tl.png", scale = 1.25)

errors_test <- read.csv("test_errors.csv")

ggplot(errors_test, aes(x=error_v, y=error_lag)) + 
  geom_point() +
  theme_gray(base_size = 23) +
  xlab("Error in Velocity") + 
  ylab("Error in Time Lag") + 
  theme(legend.position="top")

ggsave("scatter_errors_test.png", scale = 1.25)