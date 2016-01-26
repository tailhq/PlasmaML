#! /usr/bin/Rscript
library(ggplot2)
setwd("~/Development/DynaML/data/")

df <- read.csv("OmniRes.csv", 
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))

qplot(data = df, x = rsq, y = yi, facets = modelSize~stepAhead, geom = "smooth", xlab = "R sq", 
      ylab = "Model Yield")

qplot(data = df, x = modelSize, y = yi, facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Yield")

qplot(data = df, x = modelSize, y = corr, facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Correlation")

qplot(data = df, x = modelSize, y = rsq, facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Predictive Effeciency")

qplot(data = df, x = modelSize, y = mae, facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Mean Absolute Error")

qplot(data = df, x = modelSize, y = rmse, facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Root Mean Sq. Error")

qplot(data = df, x = factor(modelSize), y = yi, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Size", ylab = "Model Yield")

qplot(data = df, x = factor(modelSize), y = mae, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Size", ylab = "Model Yield")

qplot(data = df, x = factor(modelSize), y = corr, fill = factor(order),
      facets = stepAhead~., geom = "boxplot", xlab = "Model Size", ylab = "Prediction Correlation")

qplot(data = df, x = factor(modelSize), y = deltaT, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Size", ylab = "Prediction Correlation")
