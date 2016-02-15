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

dfone <- df[df$stepAhead == 1.0 ,]

dfthree <- df[df$stepAhead == 3.0 ,]

dfsix <- df[df$stepAhead == 6.0 ,]
#colorless box plots

p8 <- qplot(data = dfone, x = factor(modelSize), y = mae,
      facets = .~order, geom = "boxplot", 
      xlab = "Model Size", ylab = "Mean Abs. Error")

p9 <- qplot(data = dfone, x = factor(modelSize), y = corr,
      facets = .~order, geom = "boxplot", 
      xlab = "Model Size", ylab = "Cross Correlation")

p10 <- qplot(data = dfone, x = factor(modelSize), y = yi,
      facets = .~order, geom = "boxplot", 
      xlab = "Model Size", ylab = "Model Yield")

p1x <- qplot(data = dfone, x = factor(modelSize), y = deltaT,
             facets = .~order, geom = "boxplot", 
             xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p8, p9, p10, p1x, nrow = 2, ncol=2)

#three hours ahead


p8 <- qplot(data = dfthree, x = factor(modelSize), y = mae,
            facets = .~order, geom = "boxplot", 
            xlab = "Model Size", ylab = "Mean Abs. Error")

p9 <- qplot(data = dfthree, x = factor(modelSize), y = corr,
            facets = .~order, geom = "boxplot", 
            xlab = "Model Size", ylab = "Cross Correlation")

p10 <- qplot(data = dfthree, x = factor(modelSize), y = yi,
             facets = .~order, geom = "boxplot", 
             xlab = "Model Size", ylab = "Model Yield")

p1x <- qplot(data = dfthree, x = factor(modelSize), y = deltaT,
             facets = .~order, geom = "boxplot", 
             xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p8, p9, p10, p1x, nrow = 2, ncol=2)

#6 hours ahead


p8 <- qplot(data = dfsix, x = factor(modelSize), y = mae,
            facets = .~order, geom = "boxplot", 
            xlab = "Model Size", ylab = "Mean Abs. Error")

p9 <- qplot(data = dfsix, x = factor(modelSize), y = corr,
            facets = .~order, geom = "boxplot", 
            xlab = "Model Size", ylab = "Cross Correlation")

p10 <- qplot(data = dfsix, x = factor(modelSize), y = yi,
             facets = .~order, geom = "boxplot", 
             xlab = "Model Size", ylab = "Model Yield")

p1x <- qplot(data = dfsix, x = factor(modelSize), y = deltaT,
             facets = .~order, geom = "boxplot", 
             xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p8, p9, p10, p1x, nrow = 2, ncol=2)





#fill stepAhead

p11 <- qplot(data = df, x = factor(order), y = mae, fill = factor(stepAhead),
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Mean Abs. Error")

p12 <- qplot(data = df, x = factor(order), y = corr, fill = factor(stepAhead),
      facets = .~modelSize, geom = "boxplot", 
      xlab = "Model Order", ylab = "Cross Correlation")

p12x <- qplot(data = df, x = factor(order), y = yi, fill = factor(stepAhead),
             facets = .~modelSize, geom = "boxplot", 
             xlab = "Model Order", ylab = "Model Yield")

p12y <- qplot(data = df, x = factor(order), y = deltaT, fill = factor(stepAhead),
              facets = .~modelSize, geom = "boxplot", 
              xlab = "Model Order", ylab = "Timing Error")

grid.arrange(p11, p12, p12x, p12y, nrow = 2, ncol=2)

#fill model size
p13 <- qplot(data = df, x = factor(order), y = corr, fill = factor(modelSize),
      facets = .~stepAhead, geom = "boxplot", 
      xlab = "Model Order", ylab = "Cross Correlation")

p14 <- qplot(data = df, x = factor(order), y = deltaT, fill = factor(modelSize), 
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
      ylab = "Timing Error")

p15 <- qplot(data = df, x = factor(order), y = yi, fill = factor(modelSize), 
      facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
      ylab = "Model Yield")

p15x <- qplot(data = df, x = factor(order), y = mae, fill = factor(modelSize), 
             facets = .~stepAhead, geom = "boxplot", xlab = "Model Order", 
             ylab = "Mean Abs. Error")

grid.arrange(p15x, p13, p15, p14, nrow = 2, ncol=2)

#fill Model order

p16 <- qplot(data = df, x = factor(modelSize), y = rsq, fill = factor(stepAhead),
      facets = .~order, geom = "boxplot", xlab = "Model Size", 
      ylab = "Model Predictive Effeciency")

p17 <- qplot(data = df, x = factor(modelSize), y = yi, fill = factor(stepAhead),
      facets = .~order, geom = "boxplot",
      xlab = "Model Size", ylab = "Model Yield")

p18 <- qplot(data = df, x = factor(modelSize), y = corr, fill = factor(stepAhead),
      facets = .~order, geom = "boxplot",
      xlab = "Model Size", ylab = "Cross Correlation")

p19 <- qplot(data = df, x = factor(modelSize), y = deltaT, fill = factor(stepAhead),
      facets = .~order, geom = "boxplot", xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p16, p17, p18, p19, nrow = 2, ncol=2)