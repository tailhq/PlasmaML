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

df <- df[df$modelSize > 25,]


qplot(data = df, x = modelSize, y = yi, fill = factor(stepAhead), 
      facets = .~order, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Yield")

qplot(data = df, x = modelSize, y = corr, fill = factor(stepAhead), 
      facets = .~order, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Correlation")

qplot(data = df, x = modelSize, y = rsq, fill = factor(stepAhead), 
      facets = .~order, 
      geom = "smooth", xlab = "Model Size", 
      ylab = "Model Predictive Effeciency")

qplot(data = df, x = modelSize, y = mae, fill = factor(stepAhead),
      facets = .~order, geom = "smooth", xlab = "Model Size", 
      ylab = "Mean Absolute Error")

qplot(data = df, x = modelSize, y = rmse, fill = factor(stepAhead), 
      facets = .~order, geom = "smooth", xlab = "Model Size", 
      ylab = "Root Mean Sq. Error")

#scatter plots
qplot(data = df, x = yi, y = corr, color = factor(order), 
      facets = stepAhead~modelSize, geom = "point", xlab = "Model Yield", 
      ylab = "Model Correlation")

qplot(data = df, x = mae, y = yi, color = factor(order), facets = stepAhead~modelSize, 
      geom = "point", xlab = "Mean Abs. Error", 
      ylab = "Model Yield")



#colorless box plots
qplot(data = df, x = factor(order), y = mae,
      facets = stepAhead~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Mean Abs. Error")

qplot(data = df, x = factor(order), y = corr,
      facets = stepAhead~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Correlation")

#fill stepAhead

qplot(data = df, x = factor(order), y = mae, fill = factor(stepAhead),
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Mean Abs. Error")

qplot(data = df, x = factor(order), y = corr, fill = factor(stepAhead),
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Correlation")

#fill model size
qplot(data = df, x = factor(order), y = corr, fill = factor(modelSize),
      facets = .~stepAhead, geom = "boxplot", 
      xlab = "Model Order", ylab = "Prediction Correlation")

qplot(data = df, x = factor(order), y = rsq, fill = factor(modelSize), 
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
      ylab = "Prediction Efficiency")

qplot(data = df, x = factor(order), y = yi, fill = factor(modelSize), 
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
      ylab = "Model Yield")

#fill order

qplot(data = df, x = factor(modelSize), y = rsq, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Prediction: Steps Ahead", 
      ylab = "Model Predictive Effeciency")

qplot(data = df, x = factor(modelSize), y = yi, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot",
      xlab = "Prediction: Steps Ahead", ylab = "Model Yield")

qplot(data = df, x = factor(modelSize), y = deltaT, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Size", ylab = "Timing Error")
