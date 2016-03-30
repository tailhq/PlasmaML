#! /usr/bin/Rscript
library(plyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(latex2exp)
setwd("~/Development/DynaML/data/")

df <- read.csv("Final_NARXOmniARXStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))


df2 <- read.csv("FinalOmniARStormsRes.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("eventID","stormCat","order", "modelSize",
                              "rmse", "corr", "deltaDstMin", "DstMin",
                              "deltaT"))

dfPer <- read.csv("OmniPerStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))

dfVBz <- read.csv("Final_NARXVBzOmniARXStormsRes.csv", 
                  header = FALSE, stringsAsFactors = TRUE, 
                  col.names = c("eventID","stormCat","order", "modelSize",
                                "rmse", "corr", "deltaDstMin", "DstMin",
                                "deltaT"))

dfPer$model <- rep("Persist(1)", nrow(dfPer))

df2$model <- rep("NAR-Poly", nrow(df2))

df$model <- rep("NARX-Poly", nrow(df))

dfVBz$model <-rep("NARX-Poly-VBz", nrow(dfVBz))


bindDF <- rbind(df, df2, dfPer, dfVBz)


meltedDF <- melt(bindDF[bindDF$order == 6 | bindDF$model != "NAR-Poly",],
                 id.vars=c("model", "stormCat", "eventID"))

meltedDF1 <- meltedDF[meltedDF$variable != "order" & 
                        meltedDF$variable != "modelSize" &
                        meltedDF$variable != "DstMin",]

meansGlobal <- ddply(meltedDF1, c("model", "variable"), summarise,
               meanValue=mean(value))

meansGlobalAbs <- ddply(meltedDF1, c("model", "variable"), summarise,
                     meanValue=mean(abs(value)))

setwd("~/Development/PlasmaML/omni/data")

dfother <- read.csv("resultsModels.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("model","variable","meanValue"))

finalDF <- rbind(dfother, 
                 meansGlobal[meansGlobal$variable != "deltaT" & 
                                        meansGlobal$model %in% c("NAR-Poly", "Persist(1)",
                                                                 "NARX-Poly-VBz", "NARX-Poly"),], 
                 meansGlobalAbs[meansGlobalAbs$variable == "deltaT" & 
                                  meansGlobalAbs$model %in% c("NAR-Poly", "Persist(1)",
                                                              "NARX-Poly-VBz", "NARX-Poly"),])

finalDF$colorVal <- with(finalDF, ifelse(!(model %in% c("Persist(1)", "NAR-Poly", "NARX-Poly-VBz")), 0, 
                                     match(model, c("Persist(1)", "NAR-Poly", "NARX-Poly-VBz"))))


barplrmse1 <- ggplot(dfother[dfother$variable == "rmse",], 
                 aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 18) +
  xlab("Model") + ylab("Mean RMSE")

colourPalette <- c("0" = "grey38", "1" = "firebrick2", 
                   "2" = "steelblue3", "3" = "steelblue2")

barplrmse2 <- ggplot(finalDF[finalDF$variable == "rmse" & 
                               !(finalDF$model %in% c("NARX-Poly", "NAR-Poly", "NARX-Poly-VBz")),], 
                     aes(x = reorder(model, desc(meanValue)), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  theme_gray(base_size = 20) + 
  xlab("Model") + ylab("Mean RMSE")


barplrmse3 <- ggplot(finalDF[finalDF$variable == "rmse" & !(finalDF$model %in% c("NARX-Poly")),], 
                     aes(x = reorder(model, desc(meanValue)), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 20) +
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  xlab("Model") + ylab("Mean RMSE")



barplcc1 <- ggplot(dfother[dfother$variable == "corr",], 
                     aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 18) +
  xlab("Model") + ylab("Mean RMSE")


barplcc2 <- ggplot(finalDF[finalDF$variable == "corr" & 
                               !(finalDF$model %in% c("NARX-Poly", "NAR-Poly", "NARX-Poly-VBz")),], 
                     aes(x = reorder(model, meanValue), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 2)), size = 7, nudge_y = 1.25) + 
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  theme_gray(base_size = 20) + 
  xlab("Model") + ylab("Mean RMSE")


barplcc3 <- ggplot(finalDF[finalDF$variable == "corr" & !(finalDF$model %in% c("NARX-Poly")),], 
                     aes(x = reorder(model, meanValue), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 2)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 20) +
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  xlab("Model") + ylab("Mean RMSE")








barpl5 <- ggplot(finalDF[finalDF$variable == "deltaDstMin",], 
                 aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 18) +
  xlab("Model") + ylab(TeX('$\\Delta (D_{st})_{min}$'))

barpl6 <- ggplot(finalDF[finalDF$variable == "deltaT",], aes(x=reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 0.1) + 
  theme_gray(base_size = 18) +
  xlab("Model") + ylab("Timing Error")

barpl7 <- ggplot(finalDF[finalDF$variable == "rmse",], 
                 aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 18) +
  xlab("Model") + ylab("Mean RMSE")

barpl8 <- ggplot(finalDF[finalDF$variable == "corr",], aes(x = reorder(model, meanValue), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 2)), size = 7, nudge_y = 0.025) + 
  theme_gray(base_size = 18) + 
  xlab("Model") + ylab("Mean Corr. Coefficient")

deltaDstPlot <- ggplot(dfVBz, aes(x=DstMin, y=deltaDstMin/DstMin)) + 
  geom_point(aes(color=as.factor(stormCat))) + theme_gray(base_size = 18) + 
  labs(x = TeX('$min(D_{st})$'), 
       y=TeX('$\\frac{\\Delta D_{st}}{min(D_{st})}$'), color="Storm Category")