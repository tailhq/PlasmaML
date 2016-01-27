#! /usr/bin/Rscript
library(ggplot2)
library(gridExtra)
setwd("~/Development/DynaML/data/")

df <- read.csv("OmniNARXRes.csv", 
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "exogenousInputs","stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))


p1 <- qplot(data = df, x = modelSize, y = yi, fill = factor(order), color = factor(order),
      geom = "smooth", xlab = "Model Size", 
      ylab = "Model Yield")

p2 <- qplot(data = df, x = modelSize, y = corr, geom = "smooth", xlab = "Model Size", 
      fill = factor(order), color = factor(order), ylab = "Prediction-Output Correlation")

p3 <- qplot(data = df, x = modelSize, y = mae, geom = "smooth", xlab = "Model Size", 
      fill = factor(order), color = factor(order), ylab = "Mean Absolute Error")

p4 <- qplot(data = df, x = modelSize, y = rmse, geom = "smooth", xlab = "Model Size", 
            fill = factor(order), color = factor(order), ylab = "Root Mean Sq. Error")

grid.arrange(p1, p2, p3, p4, nrow = 2, ncol=2)


p5 <- qplot(data = df, x = order, y = rsq, geom = "smooth", xlab = "Model Order", 
      fill = factor(modelSize), color = factor(modelSize), ylab = "Prediction Efficiency")

p6 <- qplot(data = df, x = order, y = corr, geom = "smooth", xlab = "Model Order", 
      fill = factor(modelSize), color = factor(modelSize), ylab = "Prediction-Output Correlation")

p7 <- qplot(data = df, x = order, y = mae, geom = "smooth", xlab = "Model Order", 
            fill = factor(modelSize), color = factor(modelSize), ylab = "Mean Absolute Error")

p8 <- qplot(data = df, x = order, y = rmse, geom = "smooth", xlab = "Model Order", 
            fill = factor(modelSize), color = factor(modelSize), ylab = "Root Mean Sq. Error")


#colorless box plots
p9 <- qplot(data = df, x = factor(order), y = corr,
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Prediction-Output Correlation")

p10 <- qplot(data = df, x = factor(order), y = yi,
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Yield")


p11 <- qplot(data = df, x = factor(order), y = mae, geom = "boxplot", 
      facets = .~modelSize, xlab = "Model Order", ylab = "Mean Abs. Error")

p12 <- qplot(data = df, x = factor(order), y = rmse, geom = "boxplot", 
      facets = .~modelSize, xlab = "Model Order", ylab = "Root Mean Sq. Error")

grid.arrange(p9, p10, p11, p12, nrow = 2, ncol=2)
#fill model size

p13 <- qplot(data = df, x = factor(order), y = mae, geom = "boxplot", 
             fill = factor(modelSize), xlab = "Model Order", ylab = "Mean Abs. Error")

p14 <- qplot(data = df, x = factor(order), y = corr, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Order", ylab = "Prediction-Output Correlation")

p15 <- qplot(data = df, x = factor(order), y = rsq, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Order", 
      ylab = "Prediction Efficiency")

p16 <- qplot(data = df, x = factor(order), y = yi, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Order", 
      ylab = "Model Yield")

grid.arrange(p13, p14, p15, p16, nrow = 2, ncol=2)

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
