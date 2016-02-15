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
df$model <- rep("NAR", nrow(df))
df$exogenousInputs <- rep(0, nrow(df)) 
df <- df[df$modelSize > 25,]
df <- df[df$stepAhead == 1.0 ,]

dfN <- read.csv("OmniNARXRes.csv" ,
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "exogenousInputs","stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))


dfN$model <- rep("NARX", nrow(dfN))

bindDF <- rbind(df, dfN)

p9 <- qplot(data = bindDF, x = factor(modelSize), y = corr,
            facets = .~order, fill = model, geom = "boxplot", 
            xlab = "Model Size", ylab = "Cross Correlation")

p10 <- qplot(data = bindDF, x = factor(modelSize), y = yi,
             facets = .~order, fill = model, geom = "boxplot", 
             xlab = "Model Size", ylab = "Model Yield")


p11 <- qplot(data = bindDF, x = factor(modelSize), y = mae, geom = "boxplot", 
             facets = .~order, fill = model, xlab = "Model Size", ylab = "Mean Abs. Error")

p12 <- qplot(data = bindDF, x = deltaT, geom = "histogram", 
             facets = model~order, fill = model, xlab = "Model Size", ylab = "Timing Error")

grid.arrange(p11, p9, p10, p12, nrow = 2, ncol=2)
