library(plyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(latex2exp)

setwd("~/Development/PlasmaML/data")

df <- read.csv("OmniOSARes.csv", 
               header = TRUE, 
               stringsAsFactors = FALSE)

df$modelOrder = with(df, order+order_ex_1+order_ex_2)

resultsAR <- df[df$model=="GP-AR",]
resultsARX <- df[df$model=="GP-ARX",]


#first for maximum likelihood
scatterDFML <- df[df$modelOrder <= 12 & 
                    df$data == "validation" & 
                    df$modelOrder > 2 &
                    df$globalOpt == "ML",]

#then for grid search and coupled simulated annealing
scatterDF <- df[
  df$modelOrder <= 12 & 
    df$data == "validation" & 
    df$modelOrder > 2 & 
    df$globalOpt != "ML",]



bestresults <- resultsARX[
  resultsARX$modelOrder >= 6 & 
    resultsARX$modelOrder <= 10 & 
    resultsARX$data == "validation",]

bestresML <- bestresults[bestresults$globalOpt == "ML",]

bestres <- bestresults[bestresults$globalOpt != "ML",]

scaleRes <- 0.5
#plot results for GP-AR model
ggplot(resultsAR) + 
  geom_bar(stat="identity",aes(as.factor(modelOrder), mae)) + 
  facet_grid(data~globalOpt, scales = "free") +
  theme_gray(base_size = 14) + 
  xlab("Model Order") + 
  ylab("Mean Absolute Error")

#plot mae vs model order for both models, all optimization routines
ggplot(df[df$modelOrder < 36 & df$data == "validation",]) + 
  geom_boxplot(aes(as.factor(modelOrder), mae)) + 
  facet_grid(.~model, scales = "free") +
  theme_gray(base_size = 14) + 
  xlab("Model Order") + 
  ylab("Mean Absolute Error")

ggsave(
  filename = "Compare-mae.png", 
  scale = scaleRes)

#plot cc vs model order for both models, all optimization routines
ggplot(df[df$modelOrder < 36 & df$data == "validation",]) + 
  geom_boxplot(aes(as.factor(modelOrder), cc)) + 
  facet_grid(.~model, scales = "free") +
  theme_gray(base_size = 14) + 
  xlab("Model Order") + 
  ylab("Coefficient of Correlation")

ggsave(
  filename = "Compare-cc.png", 
  scale = scaleRes)


#plot mae vs model order for GP-ARX for data vs optimization routine
ggplot(resultsARX[resultsARX$data == "validation",]) + 
  geom_boxplot(aes(as.factor(modelOrder), mae)) + 
  facet_grid(.~globalOpt) +
  theme_gray(base_size = 14) + 
  xlab("Model Order") + 
  ylab("Mean Absolute Error")
ggsave(
  filename = "Compare-mae-arx.png", 
  scale = scaleRes)

#plot cc vs model order for GP-ARX for data vs optimization routine
ggplot(resultsARX[resultsARX$data == "validation",]) + 
  geom_boxplot(aes(as.factor(modelOrder), cc)) + 
  facet_grid(.~globalOpt) +
  theme_gray(base_size = 14) + 
  xlab("Model Order") + 
  ylab("Coefficient of Correlation")
ggsave(
  filename = "Compare-cc-arx.png", 
  scale = scaleRes)

#scatter plot of kernel configurations selected
ggplot(scatterDFML) + 
  geom_point(aes(
    MLPKernel.550b6925.w, 
    MLPKernel.550b6925.b, 
    color=modelOrder)) +
  facet_grid(globalOpt~model) + 
  theme_gray(base_size = 14) +
  scale_colour_gradient() +
  xlab("MLP-Kernel : w") + 
  ylab("MLP-Kernel : b")


ggplot(scatterDF) + 
  geom_point(aes(
    MLPKernel.550b6925.w, 
    MLPKernel.550b6925.b, 
    color=modelOrder)) +
  facet_grid(globalOpt~model) + 
  theme_gray(base_size = 14) +
  scale_colour_gradient() +
  xlab("MLP-Kernel : w") + 
  ylab("MLP-Kernel : b")


ggplot(bestres) + 
  geom_point(aes(
    MLPKernel.550b6925.w, 
    MLPKernel.550b6925.b, 
    color=mae)) +
  facet_grid(globalOpt~modelOrder) + 
  theme_gray(base_size = 14) +
  scale_colour_gradient() +
  xlab("MLP-Kernel : w") + 
  ylab("MLP-Kernel : b")


ggplot(bestresML) + 
  geom_point(aes(
    MLPKernel.550b6925.w, 
    MLPKernel.550b6925.w, 
    color=mae)) +
  facet_grid(.~modelOrder) + 
  theme_gray(base_size = 14) +
  scale_colour_gradient() +
  xlab("MLP-Kernel : w") + 
  ylab("MLP-Kernel : b")


#Visualise some storm predictions


palette1 <- c("#000000", "firebrick3", "forestgreen", "steelblue2")
lines1 <- c("solid", "solid", "dotdash", "dotdash")



for(i in 1:63) {
  stormName = paste("geomagnetic_storms_storm",i,".csv", sep="")
  arx_errorbars_pred <- read.csv(stormName, 
                                 header = FALSE, 
                                 col.names = c("Dst", "predicted", "lower", "upper"))
  arx_errorbars_pred$time <- 1:nrow(arx_errorbars_pred)

  meltedPred <- melt(arx_errorbars_pred, id="time")
  
  ggplot(meltedPred, aes(x=time,y=value, colour=variable, linetype=variable)) + 
    geom_point() +
    geom_line() +
    theme_gray(base_size = 22) + 
    scale_colour_manual(values=palette1) + 
    scale_linetype_manual(values = lines1, guide=FALSE) +
    xlab("Time (hours)") + ylab("Dst (nT)")
  
  ggsave(
    filename = paste("PredErrBars_Storm",i,".png", sep = ""), 
    scale = 2.0)
    
  
}



