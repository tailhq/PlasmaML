#! /usr/bin/Rscript
library(ggplot2)
library(gridExtra)
setwd("~/Development/DynaML/data/")

df <- read.csv("OmniRes.csv", 
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))

df <- df[df$modelSize > 25,]


p1 <- qplot(data = df, x = modelSize, y = yi, fill = factor(order), 
      facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Yield")

p2 <- qplot(data = df, x = modelSize, y = corr, fill = factor(order), 
      facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Model Correlation")

p3 <- qplot(data = df, x = modelSize, y = rsq, fill = factor(order), 
      facets = .~stepAhead, 
      geom = "smooth", xlab = "Model Size", 
      ylab = "Model Predictive Effeciency")

p4 <- qplot(data = df, x = modelSize, y = mae, fill = factor(order),
      facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Mean Absolute Error")

p5 <- qplot(data = df, x = modelSize, y = rmse, fill = factor(order), 
      facets = .~stepAhead, geom = "smooth", xlab = "Model Size", 
      ylab = "Root Mean Sq. Error")

grid.arrange(p1, p2, p4, p5, nrow = 2, ncol=2)
#scatter plots
p6 <- qplot(data = df, x = yi, y = corr, color = factor(order), 
      facets = stepAhead~modelSize, geom = "point", xlab = "Model Yield", 
      ylab = "Model Correlation")

p7 <- qplot(data = df, x = mae, y = yi, color = factor(order), facets = stepAhead~modelSize, 
      geom = "point", xlab = "Mean Abs. Error", 
      ylab = "Model Yield")



#colorless box plots

p8 <- qplot(data = df, x = factor(order), y = mae,
      facets = stepAhead~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Mean Abs. Error")

p9 <- qplot(data = df, x = factor(order), y = corr,
      facets = stepAhead~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Correlation")

p10 <- qplot(data = df, x = factor(order), y = yi,
      facets = stepAhead~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Yield")

p1x <- qplot(data = df, x = factor(order), y = rmse,
             facets = stepAhead~modelSize, geom = "boxplot", 
             xlab = "Model Order", ylab = "Root Mean Sq. Error")

grid.arrange(p8, p9, p10, p1x, nrow = 2, ncol=2)

#fill stepAhead

p11 <- qplot(data = df, x = factor(order), y = mae, fill = factor(stepAhead),
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Mean Abs. Error")

p12 <- qplot(data = df, x = factor(order), y = corr, fill = factor(stepAhead),
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Model Correlation")

#fill model size
p13 <- qplot(data = df, x = factor(order), y = corr, fill = factor(modelSize),
      facets = .~stepAhead, geom = "boxplot", 
      xlab = "Model Order", ylab = "Prediction-Output Correlation")

p14 <- qplot(data = df, x = factor(order), y = rsq, fill = factor(modelSize), 
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
      ylab = "Prediction Efficiency")

p15 <- qplot(data = df, x = factor(order), y = yi, fill = factor(modelSize), 
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
      ylab = "Model Yield")

p15x <- qplot(data = df, x = factor(order), y = mae, fill = factor(modelSize), 
             facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
             ylab = "Mean Abs. Error")

grid.arrange(p13, p14, p15, p15x, nrow = 2, ncol=2)

#fill Model order

p16 <- qplot(data = df, x = factor(modelSize), y = rsq, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Size", 
      ylab = "Model Predictive Effeciency")

p17 <- qplot(data = df, x = factor(modelSize), y = yi, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot",
      xlab = "Model Size", ylab = "Model Yield")

p18 <- qplot(data = df, x = factor(modelSize), y = corr, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot",
      xlab = "Model Size", ylab = "Prediction-Output Correlation")

p19 <- qplot(data = df, x = factor(modelSize), y = deltaT, fill = factor(order),
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p16, p17, p18, p19, nrow = 2, ncol=2)