#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
library(latex2exp)
library(directlabels)
library(plyr)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]

if (length(args) > 1) {
  iden <- args[2]
} else {
  ""
}


setwd(direc)

features_train_data <- read.csv("train_data_features.csv")
targets_train_data <- read.csv("train_data_targets.csv")

# Get Scaling properties
cx <- sapply(features_train_data, mean)
sx <- sapply(features_train_data, sd)

rescale_features <- function(x) {
  x*sx + cx
}

cy <- sapply(targets_train_data, mean)
sy <- sapply(targets_train_data, sd)

rescale_targets <- function(x) {
  x*sy + cy
}



train_split_features <- as.data.frame(sapply(read.csv("train_split_features.csv"), rescale_features))

x_2 <- apply(train_split_features, c(1), function(x) sum(x ^ 2))

train_split_targets <- as.data.frame(sapply(read.csv("train_split_targets.csv"), rescale_targets))

selected_preds <- Filter(function(x) x%%3 == 0, 1:ncol(train_split_targets))


pred_it_files <- list.files(pattern = "train_split_pdtit_[1-9]*_predictions.csv", recursive = FALSE)

l <- lapply(pred_it_files, read.csv)



selected_cols <- c(
  "x_2",
  sapply(selected_preds, function(x) paste(c("Target", as.character(x)), collapse = '_')),
  sapply(selected_preds, function(x) paste(c("Pred", as.character(x)), collapse = '_')),
  "it"
)

proc_preds <- function(index) {
  df <- as.data.frame(sapply(l[[index]], rescale_targets))
  df$it <- rep(index, nrow(df))
  proc <- cbind(x_2, train_split_targets, df)
  colnames(proc) <- c(
    "x_2", 
    lapply(1:ncol(train_split_targets), function(i){paste(c("Target", i - 1), collapse = '_')}),
    lapply(1:ncol(train_split_targets), function(i){paste(c("Pred", i - 1), collapse = '_')}),
    "it"
    )
  proc[,as.vector(selected_cols)]
}

train_preds_and_targets_by_it <- ldply(lapply(seq_along(l), proc_preds), data.frame)

scatter_predictors <- melt(
  train_preds_and_targets_by_it,
  id = c("x_2", "it")
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

ggplot(
  scatter_predictors, 
  aes(x = x_2, y = value, colour = variable)) +
  geom_point(alpha = 0.65) +
  theme_gray(base_size = 14) + 
  scale_colour_manual(values=palette1, labels = unname(TeX(c('$\\hat{y}(\\mathbf{x})$', '$y(\\mathbf{x})$')))) +
  facet_grid(it~timelag) +
  xlab(TeX('$||\\mathbf{x}||_2$')) + ylab(TeX('$\\hat{y}(\\mathbf{x}) \\ y(\\mathbf{x})$'))

ggsave(paste(iden, "predictors_scatter_it.pdf", sep = ''), scale = 2.5, device = pdf())


