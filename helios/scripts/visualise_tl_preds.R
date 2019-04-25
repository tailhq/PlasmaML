#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
library(latex2exp)
library(directlabels)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]

if (length(args) > 1) {
  iden <- args[2]
} else {
  ""
}


setwd(direc)

features_test <- read.csv("test_data_features.csv")
preds_test <- read.csv("test_predictions.csv")
targets_test <- read.csv("test_data_targets.csv")
errors_test <- abs(preds_test - targets_test)

probs_test <- read.csv("test_probabilities.csv")
scatter_test <- read.csv("test_scatter.csv")
output_mapping <- read.csv("output_mapping_test.csv")

size_window <- ncol(preds_test)

max_timelag <- min(size_window, max(scatter_test$actuallag))

aug_preds_test <- cbind(preds_test, output_mapping)

colnames(aug_preds_test) <- c(
  lapply(
    1:size_window,
    function(x) paste(c("Pred", x - 1), collapse = '_')
    ),
  "Output"
  )

colnames(errors_test) <- lapply(
  1:size_window,
  function(x) paste(c("Error", x - 1), collapse = '_')
)

colnames(probs_test) <- lapply(
  1:size_window,
  function(x) paste(c("Probability", x - 1), collapse = '_')
)

colnames(preds_test) <- lapply(
  1:size_window,
  function(x) paste(c("Pred", x - 1), collapse = '_')
)

colnames(targets_test) <- lapply(
  1:size_window,
  function(x) paste(c("Target", x - 1), collapse = '_')
)


x_2 <- apply(features_test, c(1), function(x) sum(x ^ 2))

prob_df <- melt(
  cbind(
    x_2, 
    probs_test[,1:max_timelag]
  ), 
  id = "x_2")

pred_df <- melt(
  cbind(x_2, aug_preds_test[size_window + 1], aug_preds_test[, 1:max_timelag]),
  id = "x_2")

err_df <- melt(
  cbind(x_2, errors_test[, 1:max_timelag]), id = "x_2"
)

scatter_predictors <- melt(
  cbind(x_2, targets_test, preds_test),
  id = "x_2"
)

scatter_predictors$timelag <- as.factor(
  unlist(
    lapply(
      as.character(scatter_predictors$variable),
      function(v) as.integer(strsplit(v, '_')[[1]][2])
    )
  )
)

scatter_predictors$variable <- as.factor(
  unlist(
    lapply(
      as.character(scatter_predictors$variable),
      function(v) as.character(strsplit(v, '_')[[1]][1])
    )
  )
)

palette1 <- c("#000000", "firebrick3")

ggplot(as.data.frame(x_2), aes(x = x_2)) + 
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 20) + xlab(TeX('$||\\mathbf{x}||_2$'))

ggsave(paste(iden, "density.pdf", sep = ''), scale = 2.0, device = pdf())

ggplot(scatter_predictors, aes(x = x_2, y = value, colour = variable)) +
  geom_point(alpha = 0.65) +
  theme_gray(base_size = 16) + 
  scale_colour_manual(values=palette1, labels = unname(TeX(c('$\\hat{y}(\\mathbf{x})$', '$y(\\mathbf{x})$')))) +
  facet_wrap(~timelag) +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$\\hat{y}(\\mathbf{x}) \\ y(\\mathbf{x})$'))

ggsave(paste(iden, "predictors_scatter.pdf", sep = ''), scale = 2.0, device = pdf())

ggplot(pred_df, aes(x = x_2, y = value, colour = variable)) +
  geom_smooth(se = FALSE) +
  theme_gray(base_size = 20) +
  scale_colour_viridis_d(labels =
    lapply(
      1:(max_timelag + 1),
      function(i) {
        if (i > 1) TeX(paste(c('$\\hat{y}_{', i - 1, '}(\\mathbf{x})$'), collapse = ''))
        else TeX(paste('$y(\\mathbf{x})$'))
      })) +
  labs(color = "Predictors") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$\\hat{y}(\\mathbf{x})$'))

ggsave(paste(iden, "predictors.pdf", sep = ''), scale = 1.25, device = pdf())

ggplot(err_df, aes(x = x_2, y = value, colour = variable)) +
  geom_smooth(se = FALSE) +
  theme_gray(base_size = 20) +
  scale_colour_viridis_d(labels =
    lapply(
      1:(max_timelag),
      function(i) {
        TeX(paste(c('$|\\hat{y}_{', i - 1, '}(\\mathbf{x}) - y_{', i - 1, '}|$'), collapse = ''))
      })) +
  labs(color = "Errors") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$|\\hat{y}_{i}(\\mathbf{x}) - y_{i}|$'))

ggsave(paste(iden, "error_curves.pdf", sep = ''), scale = 1.25, device = pdf())



ggplot(prob_df, aes(x = x_2, y = value, colour = variable)) +
  geom_smooth(se = TRUE) +
  theme_gray(base_size = 20) +
  scale_colour_viridis_d(labels = lapply(
    1:max_timelag,
    function(x) TeX(paste(c('$\\mathbf{P}(\\Delta t = ', x, ')$'), collapse = ''))
  )) + labs(color = "Time Lag Probabilities") +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab("Probability")

ggsave(paste(iden, "probabilities.pdf", sep = ''), scale = 1.25, device = pdf())







