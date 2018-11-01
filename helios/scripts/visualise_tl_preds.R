#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
library(latex2exp)
library(directlabels)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
setwd(direc)

palette1 <- c("firebrick3", "gray27", "forestgreen")
lines1 <- c("solid", "solid", "dotdash", "dotdash")

features_test <- read.csv("test_data_features.csv")
preds_test    <- read.csv("test_predictions.csv")
probs_test    <- read.csv("test_probabilities.csv")
scatter_test  <- read.csv("test_scatter.csv")
size_window   <- ncol(preds_test) 

max_timelag   <- max(scatter_test$actuallag)

colnames(preds_test) <- lapply(
  1:size_window,
  function(x) paste(c("Pred", x), collapse = '_')
)

colnames(probs_test) <- lapply(
  1:size_window,
  function(x) paste(c("Probability", x), collapse = '_')
)


x_2 <- apply(features_test, c(1), function(x) sum(x^2))

prob_df <- melt(cbind(x_2, probs_test[,1:max_timelag]), id = "x_2")

pred_df <- melt(cbind(x_2, preds_test[,1:max_timelag]), id = "x_2")

ggplot(prob_df, aes(x=x_2,y=value, colour=variable)) +
  geom_smooth(se = TRUE) +
  theme_gray(base_size = 22) + 
  scale_color_discrete(labels = lapply(
    1:max_timelag,
    function(x) TeX(paste(c('$\\mathbf{P}(\\Delta t = ', x,')$'), collapse = ''))
  )) + labs(color = "Time Lag Probabilities") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab("Probability")

ggsave("probabilities.png", scale = 1.25)

ggplot(pred_df, aes(x=x_2,y=value, colour=variable)) +
  geom_smooth(se = TRUE) +
  theme_gray(base_size = 22) + 
  scale_color_discrete(labels = lapply(
    1:max_timelag,
    function(i) TeX(paste(c('$\\hat{y}_{', i, '}(\\mathbf{x})$'), collapse = ''))
  )) + 
  labs(color = "Predictors") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$\\hat{y}(\\mathbf{x})$'))

ggsave("predictors.png", scale = 1.25)

