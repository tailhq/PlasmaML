#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]

if(length(args) > 1) {
  iden <- args[2]
} else {
  ""
}

setwd(direc)

palette1 <- c("firebrick3", "gray27", "forestgreen")
lines1 <- c("solid", "solid", "dotdash", "dotdash")

scatter_test <- read.csv("test_scatter.csv")
scatter_train <- read.csv("train_scatter.csv")

proc_scatter_test <- scatter_test[1625:1750,c(1,3)]
proc_scatter_test$time <- 1:nrow(proc_scatter_test)

meltDF <- melt(proc_scatter_test, id = "time")

p1 <- ggplot(meltDF, aes(x=time, y=value, color=variable)) + 
  geom_line() +
  scale_colour_manual(labels = c("Prediction", "Actual"), values=palette1) +
  theme_gray(base_size = 20)

ggsave(paste(iden, "timeseries_pred.pdf", sep = ''), p1, scale = 1.0, device = pdf())

scatter_df_pred <- data.frame(
  c(scatter_test$predv, scatter_train$predv), 
  c(scatter_test$predlag, scatter_train$predlag), 
  rep("predicted", length(scatter_test$predv) + length(scatter_train$predv)), 
  c(rep("test", length(scatter_test$predv)), rep("training", length(scatter_train$predv)))
  )

colnames(scatter_df_pred) <- c("Velocity", "TimeLag", "Type", "data")

scatter_df_actual <- data.frame(
  c(scatter_test$actualv, scatter_train$actualv), 
  c(scatter_test$actuallag, scatter_train$actuallag), 
  rep("actual", length(scatter_test$actualv) + length(scatter_train$actualv)),
  c(rep("test", length(scatter_test$actualv)), rep("training", length(scatter_train$actualv)))
  )

colnames(scatter_df_actual) <- c("Velocity", "TimeLag", "Type", "data")

scatter_df <- rbind(scatter_df_pred, scatter_df_actual)

colnames(scatter_df) <- c("Velocity", "TimeLag", "Type", "data")

scatter_v_test <- data.frame(scatter_test$predv, scatter_test$actualv)
colnames(scatter_v_test) <- c("predv", "actualv")

scatter_t_test <- data.frame(scatter_test$predlag, scatter_test$actuallag)
colnames(scatter_t_test) <- c("predlag", "actuallag")

ggplot(scatter_v_test, aes(x = actualv, y=predv)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method=lm) +
  theme_gray(base_size = 20) + 
  labs(y="Predicted Output", x=" Actual Output") + 
  theme(legend.position="top")

ggsave(paste(iden, "scatter_v_test.pdf", sep = ''), scale = 1.0, device = pdf())

ggplot(scatter_t_test, aes(x = actuallag, y=predlag)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method=lm) +
  theme_gray(base_size = 20) + 
  labs(y="Predicted Time Lag", x="Actual Time Lag") + 
  theme(legend.position="top")

ggsave(paste(iden, "scatter_t_test.pdf", sep = ''), scale = 1.0, device = pdf())


#Now construct the error data frames
errors_test <- read.csv("test_errors.csv")

errors_test <- cbind(errors_test, rep("test", length(errors_test$error_v)))

colnames(errors_test) <- c("error_v", "error_lag", "data")

errors_train <- read.csv("train_errors.csv")

errors_train <- cbind(errors_train, rep("train", length(errors_train$error_v)))

colnames(errors_train) <- c("error_v", "error_lag", "data")

errors <- rbind(errors_train, errors_test)


ggplot(scatter_df, aes(x = Velocity, y = floor(TimeLag), color = Type, fill = Type)) +
  geom_point(alpha = 0.5) + 
  geom_smooth(method = lm, formula = y ~ splines::bs(x, 3)) +
  facet_grid(data ~ .) +
  theme_gray(base_size = 20) + 
  scale_colour_manual(values=palette1) + 
  labs(y="Time Lag", x="Output") + 
  theme(legend.position="top")

ggsave(paste(iden, "scatter_v_tl.pdf", sep = ''), scale = 1.0, device = pdf())

#rbind(errors_train, errors_test)
ggplot(errors_test, aes(x=error_v, y=floor(error_lag))) + 
  scale_alpha_continuous(limits = c(0, 0.2), breaks = seq(0, 0.2, by=0.025)) +
  geom_point(aes(alpha = 0.05), show.legend = FALSE) + 
  #stat_density2d(aes(color = "white"), show.legend = FALSE) + 
  #theme(legend.text=element_text(size=9), legend.position = "right") +
  #facet_grid(data ~ .) +
  theme_gray(base_size = 20) +
  xlab("Error in Output") + 
  ylab("Error in Time Lag") + 
  theme(legend.position="top")

ggsave(paste(iden, "scatter_errors.pdf", sep = ''), scale = 1.0, device = pdf())

ggplot(errors_test, aes(x=error_v, y=error_lag)) + 
  #scale_alpha_continuous(limits = c(0, 0.2), breaks = seq(0, 0.2, by=0.025)) +
  #geom_point(aes(alpha = 0.2), show.legend = FALSE) + 
  stat_density_2d(aes(fill = stat(level)), geom = "polygon", show.legend = TRUE) +
  scale_fill_viridis_c(
    guide = guide_colourbar(
      direction = "horizontal",
      title.position = "top",
      label.position = "bottom",
      label.vjust = -0.001,
      label.theme = element_text(angle = 90, size = 10)
  )) +  
  #theme(legend.text=element_text(size=9), legend.position = "right") +
  #facet_grid(data ~ .) +
  theme_gray(base_size = 20) +
  xlab("Error in Output") + 
  ylab("Error in Time Lag") + 
  theme(legend.position="top")

ggsave(paste(iden, "errors.pdf", sep = ''), scale = 1.0, device = pdf())


sample_df <- scatter_test[scatter_test$predlag - scatter_test$actuallag <= -2,]

p <- ggplot(sample_df, aes(x = predv, y = actualv)) +
  geom_point(aes(alpha = predlag - actuallag)) + 
  scale_alpha(range = c(1.0, 0.1)) +
  theme_gray(base_size = 20) + 
  labs(y = "Actual Output", x = "Predicted Output", alpha = "Error: Time Lag")

ggsave(paste(iden, "lag_error_jus.pdf", sep = ''), p, scale = 1.0, device = pdf())

ggplot(errors[errors$data == "test",], aes(x=error_lag)) +
  geom_histogram(
    aes(y=..density..),      # Histogram with density instead of count on y-axis
    binwidth=.5,
     colour="black", fill="white") +
  #geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 20) +
  xlab("Error: Time Lag") #+ facet_grid(data ~ .)

ggsave(paste(iden, "hist_errors_timelag.pdf", sep = ''), scale = 1.0, device = pdf())


ggplot(scatter_df[scatter_df$data == "test",], aes(x = Velocity, y = floor(TimeLag), color = Type, fill = Type)) +
  #geom_point(alpha = 0.5) + 
  geom_smooth(method = lm, formula = y ~ splines::bs(x, 3)) +
  #facet_grid(data ~ .) +
  theme_gray(base_size = 20) + 
  scale_colour_manual(values=palette1) + 
  labs(y="Time Lag", x="Output") + 
  theme(legend.position="top")

ggsave(paste(iden, "predictive_curves.pdf", sep = ''), scale = 1.0, device = pdf())

#Construct loess/lm model

# invVModel <- 1/scatter_df_pred$Velocity
# vModel <- scatter_df_pred$Velocity
# lag_model <- scatter_df_pred$TimeLag
# 
# model <- loess(lag_model ~ invVModel, family = "symmetric")
# 
# 
# invVActual <- 1/scatter_df_actual$Velocity
# vActual <- scatter_df_actual$Velocity
# lag_actual <- scatter_df_actual$TimeLag
# 
# actual <- loess(lag_actual ~ invVActual, family = "symmetric")
# 
# model_preds <- predict(model)
# actual_preds <- predict(actual)
# model_df <- data.frame(vModel, model_preds, rep("model", length(model_preds)))
# colnames(model_df) <- c("Velocity", "TimeLag", "Type")
# 
# 
# actual_df <- data.frame(vActual, actual_preds, rep("actual", length(actual_preds)))
# colnames(actual_df) <- c("Velocity", "TimeLag", "Type")
# 
# df <- as.data.frame(rbind(model_df, actual_df))
# 
# ggplot(df, aes(x = Velocity, y = TimeLag, color = Type)) + 
#   geom_smooth(size=1.25) +
#   theme_gray(base_size = 20) + 
#   scale_colour_manual(values=palette1) + 
#   labs(y="Time Lag", x="Output") + 
#   theme(legend.position="top")
# 
# ggsave(paste(iden, 'predictive_curves.pdf', sep = ''), scale = 1.0, device = pdf())
