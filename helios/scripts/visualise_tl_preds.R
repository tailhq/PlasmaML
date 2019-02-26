#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
library(latex2exp)
library(directlabels)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
setwd(direc)

features_test <- read.csv("test_data_features.csv")
preds_test    <- read.csv("test_predictions.csv")
targets_test  <- read.csv("test_data_targets.csv")
errors_test   <- abs(preds_test - targets_test)

probs_test    <- read.csv("test_probabilities.csv")
scatter_test  <- read.csv("test_scatter.csv")
output_mapping <- read.csv("output_mapping_test.csv")

size_window   <- ncol(preds_test) 

max_timelag   <- max(scatter_test$actuallag)

aug_preds_test <- cbind(preds_test, output_mapping)

colnames(aug_preds_test) <- c(
  lapply(
    1:size_window, 
    function(x) paste(c("Pred", x-1), collapse = '_')
    ), 
  "Output"
  )

colnames(errors_test) <- lapply(
  1:size_window, 
  function(x) paste(c("Error", x-1), collapse = '_')
)

colnames(probs_test) <- lapply(
  1:size_window,
  function(x) paste(c("Probability", x-1), collapse = '_')
)


x_2 <- apply(features_test, c(1), function(x) sum(x^2))

prob_df <- melt(cbind(x_2, probs_test[,1:max_timelag]), id = "x_2")

pred_df <- melt(
  cbind(x_2, aug_preds_test[size_window+1], aug_preds_test[,1:max_timelag]), 
  id = "x_2")

err_df <- melt(
  cbind(x_2, errors_test[,1:max_timelag]), id = "x_2"
)

ggplot(pred_df, aes(x=x_2, y=value, colour=variable)) +
  geom_smooth(se = FALSE) +
  theme_gray(base_size = 22) + 
  scale_colour_viridis_d(labels = 
    lapply(
      1:(max_timelag+1),
      function(i) {
        if(i > 1) TeX(paste(c('$\\hat{y}_{', i-1, '}(\\mathbf{x})$'), collapse = '')) 
        else TeX(paste('$y(\\mathbf{x})$'))
      })) + 
  labs(color = "Predictors") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$\\hat{y}(\\mathbf{x})$'))

ggsave("predictors.png", scale = 1.25)

ggplot(err_df, aes(x=x_2, y=value, colour=variable)) +
  geom_smooth(se = FALSE) +
  theme_gray(base_size = 22) + 
  scale_colour_viridis_d(labels = 
    lapply(
      1:(max_timelag),
      function(i) {
        TeX(paste(c('$|\\hat{y}_{', i-1, '}(\\mathbf{x}) - y_{',i-1,'}|$'), collapse = ''))
      })) + 
  labs(color = "Errors") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$|\\hat{y}_{i}(\\mathbf{x}) - y_{i}|$'))

ggsave("errors.png", scale = 1.25)



ggplot(prob_df, aes(x=x_2,y=value, colour=variable)) +
  geom_smooth(se = TRUE) +
  theme_gray(base_size = 22) + 
  scale_colour_viridis_d(labels = lapply(
    1:max_timelag,
    function(x) TeX(paste(c('$\\mathbf{P}(\\Delta t = ', x,')$'), collapse = ''))
  )) + labs(color = "Time Lag Probabilities") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab("Probability")

ggsave("probabilities.png", scale = 1.25)
