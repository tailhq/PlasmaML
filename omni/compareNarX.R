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

df$exogenousInputs <- rep(0, nrow(df)) 
df <- df[df$modelSize > 25,]

dfthree <- df[df$stepAhead == 3.0 ,]
dfone <- df[df$stepAhead == 1.0 ,]

dfone$model <- rep("NAR-1", nrow(dfone))
dfthree$model <- rep("NAR-3", nrow(dfthree))

dfN <- read.csv("OmniNARXRes.csv" ,
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",12), 
               col.names = c("trainyear", "testyear", 
                             "order", "exogenousInputs","stepAhead",
                             "modelSize", "numTest", "mae", "rmse", 
                             "rsq", "corr", "yi", "deltaT"))

dfN$model <- rep("NARX", nrow(dfN))

bindDF <- rbind(dfone, dfthree, dfN)

p9 <- qplot(data = bindDF, x = factor(modelSize), y = corr,
            facets = .~order, fill = model, geom = "boxplot", 
            xlab = "Model Size", ylab = "Cross Correlation") + theme_grey(base_size = 18) 

p10 <- qplot(data = bindDF, x = factor(modelSize), y = yi,
             facets = .~order, fill = model, geom = "boxplot", 
             xlab = "Model Size", ylab = "Model Yield") + theme_grey(base_size = 18)

p11 <- qplot(data = bindDF, x = factor(modelSize), y = mae, geom = "boxplot", 
             facets = .~order, fill = model, xlab = "Model Size", 
             ylab = "Mean Abs. Error")  + theme_grey(base_size = 18)

p12 <- qplot(data = bindDF, x = deltaT, geom = "histogram", 
             facets = modelSize~order, fill = model, 
             xlab = "Timing Error", ylab = "Frequency")  + 
  theme_grey(base_size = 18)

prsq <- qplot(data = bindDF, x = factor(modelSize), y = rsq, geom = "boxplot", 
             facets = .~order, fill = model, xlab = "Model Size", 
             ylab = "R SQ")  + theme_grey(base_size = 18)

grid.arrange(p11, p9, p10, p12, nrow = 2, ncol=2)