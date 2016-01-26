#! /usr/bin/Rscript
library(ggplot2)
setwd("~/Development/DynaML/data/")

df <- read.csv("OmniNARXRes.csv", 
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "exogenousInputs","stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))


qplot(data = df, x = modelSize, y = yi, fill = factor(order), color = factor(order),
      geom = "smooth", xlab = "Model Order", 
      ylab = "Model Yield")

qplot(data = df, x = modelSize, y = corr, geom = "smooth", xlab = "Model Size", 
      facets = .~order, ylab = "Model Correlation")

qplot(data = df, x = modelSize, y = rsq, geom = "smooth", xlab = "Model Size", 
      facets = .~order, ylab = "Predictive Efficiency")

qplot(data = df, x = modelSize, y = mae, geom = "smooth", xlab = "Model Size", 
      fill = factor(order), color = factor(order), ylab = "Mean Absolute Error")

qplot(data = df, x = modelSize, y = rmse, geom = "smooth", xlab = "Model Size", 
      facets = .~order, ylab = "Root Mean Sq. Error")

qplot(data = df, x = order, y = rsq, geom = "smooth", xlab = "Model Order", 
      ylab = "Prediction Efficiency")


#colorless box plots
qplot(data = df, x = factor(order), y = mae, geom = "boxplot", 
      facets = .~modelSize,xlab = "Model Order", ylab = "Mean Abs. Error")

qplot(data = df, x = factor(order), y = corr,
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Correlation")

qplot(data = df, x = factor(order), y = yi,
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Yield")

#fill model size
qplot(data = df, x = factor(order), y = corr, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Order", ylab = "Prediction-Output Correlation")

qplot(data = df, x = factor(order), y = rsq, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Order", 
      ylab = "Prediction Efficiency")

qplot(data = df, x = factor(order), y = yi, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Order", 
      ylab = "Model Yield")

#fill Model order

qplot(data = df, x = factor(modelSize), y = rsq, fill = factor(order),
      geom = "boxplot", xlab = "Model Size", 
      ylab = "Model Predictive Effeciency")

qplot(data = df, x = factor(modelSize), y = yi, fill = factor(order),
      geom = "boxplot",
      xlab = "Model Size", ylab = "Model Yield")

qplot(data = df, x = factor(modelSize), y = corr, fill = factor(order),
      geom = "boxplot",
      xlab = "Model Size", ylab = "Prediction-Output Correlation")

qplot(data = df, x = factor(modelSize), y = deltaT, facets = .~order,
      geom = "boxplot", xlab = "Model Size", ylab = "Timing Error")
