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

features_train_data <- read.csv("train_data_features.csv")
targets_train_data <- read.csv("train_data_targets.csv")


train_split_features <- read.csv("train_split_features.csv")
train_split_targets <- read.csv("train_split_targets.csv")

load_preds <- function(data, index) {
  df <- read.csv(data)
  df$it <- rep(index, nrow(df))
}

pred_it_files <- list.files(pattern = "train_split_pdtit_[1-9]*_predictions.csv", recursive = FALSE)

train_preds_by_it <- mapply(load_preds, pred_it_files, 1:length(pred_it_files))
