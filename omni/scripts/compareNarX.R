#! /usr/bin/Rscript
library(ggplot2)
library(gridExtra)
library(ggthemes)
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
bindDF <- bindDF[bindDF$modelSize %in% c(100, 150, 200),]

order_labeller <- function(variable,value){
  if(is.numeric(value) && value < 100){
    return(paste("AR(",value,")"))  
  } else {
    value
  }
  
}

p9 <- qplot(data = bindDF, x = factor(model), y = corr,
             geom = "boxplot", 
            xlab = "Model", 
            ylab = "Cross Correlation", color = factor(modelSize)) + 
  theme_gray(base_size = 14) + scale_colour_tableau("colorblind10", name="Size of Training Data")+ 
  facet_grid(.~order, labeller = order_labeller)

p10 <- qplot(data = bindDF, x = factor(model), y = yi,
             geom = "boxplot", 
             xlab = "Model", ylab = "Model Yield",color = factor(modelSize)) + 
  theme_gray(base_size = 14)+
  scale_colour_tableau("colorblind10", name="Size of Training Data")+ 
  facet_grid(.~order, labeller = order_labeller)

p11 <- qplot(data = bindDF, x = factor(model), y = mae, geom = "boxplot", 
             xlab = "Model", 
             ylab = "Mean Abs. Error",color = factor(modelSize))  + 
  theme_gray(base_size = 14)+ 
  scale_colour_tableau("colorblind10", name="Size of Training Data")+
  facet_grid(.~order, labeller = order_labeller)

p12 <- qplot(data = bindDF, x = deltaT, geom = "histogram", 
             xlab = "Timing Error", ylab = "Frequency")  + 
  theme_gray(base_size = 16) + 
  facet_grid(model~order, labeller = order_labeller)

prsq <- qplot(data = bindDF, x = factor(model), y = rsq, geom = "boxplot", 
              xlab = "Model", 
             ylab = bquote(R^2), 
             color = factor(modelSize))  + 
  theme_gray(base_size = 16) + 
  scale_colour_tableau("colorblind10", name="Size of Training Data")+
  facet_grid(.~order, labeller = order_labeller)

grid.arrange(p11, p9, p10, prsq, nrow = 2, ncol=2)



p1 <- qplot(data = bindDF, x = factor(model), y = corr,
            geom = "boxplot", 
            xlab = "Model", 
            ylab = "Cross Correlation") + 
  theme_gray(base_size = 18)+
  facet_grid(.~order, labeller = order_labeller)

p2 <- qplot(data = bindDF, x = factor(model), y = yi,
             geom = "boxplot", 
             xlab = "Model", ylab = "Model Yield") + 
  theme_gray(base_size = 18)+
  facet_grid(.~order, labeller = order_labeller)

p3 <- qplot(data = bindDF, x = factor(model), y = mae, geom = "boxplot", 
             xlab = "Model", 
             ylab = "Mean Abs. Error")  + 
  theme_gray(base_size = 18)+ 
  facet_grid(.~order, labeller = order_labeller)


p4 <- qplot(data = bindDF, x = factor(model), y = rsq, geom = "boxplot", 
              xlab = "Model", 
              ylab = bquote(R^2))  + 
  theme_gray(base_size = 18) + 
  facet_grid(.~order, labeller = order_labeller)

grid.arrange(p1, p2, p3, p4, nrow = 2, ncol=2)