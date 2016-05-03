#! /usr/bin/Rscript
library(ggplot2)
library(gridExtra)
setwd("../../../DynaML/data/")

df <- read.csv("OmniNARXRes.csv" ,
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "exogenousInputs","stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))


#colorless box plots
p9 <- qplot(data = df, x = factor(modelSize), y = corr,
      facets = .~order, geom = "boxplot", 
      xlab = "Model Size", ylab = "Cross Correlation")

p10 <- qplot(data = df, x = factor(modelSize), y = yi,
      facets = .~order, geom = "boxplot", 
      xlab = "Model Size", ylab = "Model Yield")


p11 <- qplot(data = df, x = factor(modelSize), y = mae, geom = "boxplot", 
      facets = .~order, xlab = "Model Size", ylab = "Mean Abs. Error")

p12 <- qplot(data = df, x = factor(modelSize), y = deltaT, geom = "boxplot", 
      facets = .~order, xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p11, p9, p10, p12, nrow = 2, ncol=2)

#fill Model Size

p17 <- qplot(data = df, x = factor(order), y = mae, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Size", 
      ylab = "Mean Abs. Error")

p18 <- qplot(data = df, x = factor(order), y = corr, fill = factor(modelSize),
      geom = "boxplot",
      xlab = "Model Size", ylab = "Cross Correlation")

p19 <- qplot(data = df, x = factor(order), y = yi, fill = factor(modelSize),
      geom = "boxplot",
      xlab = "Model Size", ylab = "Model Yield")

p20 <- qplot(data = df, x = factor(order), y = deltaT, fill = factor(modelSize),
      geom = "boxplot", xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p17, p18, p19, p20, nrow = 2, ncol=2)
