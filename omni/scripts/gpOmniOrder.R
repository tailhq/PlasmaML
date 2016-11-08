library(plyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(latex2exp)


df <- read.csv("Nov_4_2016_alt_7_1_OmniARXStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))


df1 <- read.csv("Nov_4_2016_alt_6_1_OmniARXStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))

df2 <- read.csv("Nov_4_2016_alt_5_1_OmniARXStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))

df3 <- read.csv("Nov_4_2016_alt_4_1_OmniARXStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))

df4 <- read.csv("Nov_4_2016_alt_3_1_OmniARXStormsRes.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("eventID","stormCat","order", "modelSize",
                              "rmse", "corr", "deltaDstMin", "DstMin",
                              "deltaT"))

df5 <- read.csv("Nov_4_2016_alt_2_1_OmniARXStormsRes.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("eventID","stormCat","order", "modelSize",
                              "rmse", "corr", "deltaDstMin", "DstMin",
                              "deltaT"))

dfFinal <- rbind(df, df1, df2, df3, df4)

qplot(data = dfFinal, x = factor(order), y = rmse,
      geom = "boxplot", 
      xlab = "Model Order", ylab = "Model RMSE") + 
  theme_gray(base_size = 22)

ggsave(filename = "Model_RMSE_validationStorms.png", scale = 2.0)