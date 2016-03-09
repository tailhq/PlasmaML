#! /usr/bin/Rscript
library(ggplot2)
library(gridExtra)
library(plyr)
library(reshape2)
setwd("~/Development/DynaML/data/")

df <- read.csv("PolyOmniARXStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))


df2 <- read.csv("PolyNAROmniARStormsRes.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("eventID","stormCat","order", "modelSize",
                              "rmse", "corr", "deltaDstMin", "DstMin",
                              "deltaT"))

dfPer <- read.csv("OmniPerStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))

dfVBz <- read.csv("PolyVBzDst_OmniARXStormsRes.csv", 
                  header = FALSE, stringsAsFactors = TRUE, 
                  col.names = c("eventID","stormCat","order", "modelSize",
                                "rmse", "corr", "deltaDstMin", "DstMin",
                                "deltaT"))

dfPVBz <- read.csv("PolyPVBzDst_OmniARXStormsRes.csv", 
                  header = FALSE, stringsAsFactors = TRUE, 
                  col.names = c("eventID","stormCat","order", "modelSize",
                                "rmse", "corr", "deltaDstMin", "DstMin",
                                "deltaT"))

dfPer$model <- rep("Persist(1)", nrow(dfPer))

df2$model <- rep("NAR-Poly", nrow(df2))

df$model <- rep("NARX-Poly", nrow(df))

dfVBz$model <-rep("NARX-Poly-VBz", nrow(dfVBz))

dfPVBz$model <- rep("NARX-Poly-PVBz", nrow(dfPVBz))

bindDF <- rbind(df, df2, dfPer, dfVBz, dfPVBz)


meltedDF <- melt(bindDF[(bindDF$order == 4 | bindDF$model == "Persist(1)") & bindDF$model != "NAR-FBM",],
                 id.vars=c("model", "stormCat", "eventID"))

meltedDF1 <- meltedDF[meltedDF$variable != "order" & 
                        meltedDF$variable != "modelSize" &
                        meltedDF$variable != "DstMin",]

means <- ddply(meltedDF1, c("model", "stormCat", "variable"), summarise,
               meanValue=mean(value))

barpl <- ggplot(means[means$variable == "corr",], aes(x=model, y=meanValue, fill=stormCat)) + 
  geom_bar(stat="identity", position="dodge")


barpl1 <- ggplot(means[means$variable == "rmse",], aes(x=model, y=meanValue, fill=stormCat)) + 
  geom_bar(stat="identity", position="dodge")


barpl2 <- ggplot(means[means$variable == "deltaDstMin",], aes(x=model, y=meanValue, fill=stormCat)) + 
  geom_bar(stat="identity", position="dodge")

meansGlobal <- ddply(meltedDF1, c("model", "variable"), summarise,
               meanValue=mean(value))

barpl3 <- ggplot(meansGlobal[meansGlobal$variable == "corr",], aes(x=model, y=meanValue, fill=model)) + 
  geom_bar(stat="identity", position="dodge") + geom_text(aes(label = round(meanValue, digits = 3)))


barpl4 <- ggplot(meansGlobal[meansGlobal$variable == "rmse",], aes(x=model, y=meanValue, fill=model)) + 
  geom_bar(stat="identity", position="dodge") + geom_text(aes(label = round(meanValue, digits = 3)))


meansGlobalAbs <- ddply(meltedDF1, c("model", "variable"), summarise,
                     meanValue=mean(abs(value)))

setwd("~/Development/PlasmaML/omni/data")

dfother <- read.csv("resultsModels.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("model","variable","meanValue"))

finalDF <- rbind(dfother, 
                 meansGlobal[meansGlobal$variable != "deltaT" & 
                                        meansGlobal$model %in% c("NAR-Poly", "Persist(1)",
                                                                 "NARX-Poly-VBz", "NARX-Poly-PVBz"),], 
                 meansGlobalAbs[meansGlobalAbs$variable == "deltaT" & 
                                  meansGlobalAbs$model %in% c("NAR-Poly", "Persist(1)",
                                                              "NARX-Poly-VBz", "NARX-Poly-PVBz"),])

barpl5 <- ggplot(finalDF[finalDF$variable == "deltaDstMin",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3)), size = 7) + 
  theme_gray(base_size = 14) +
  xlab("Model") + ylab("delta(Dst min)")

barpl6 <- ggplot(finalDF[finalDF$variable == "deltaT",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3)), size = 7) + 
  theme_gray(base_size = 14) +
  xlab("Model") + ylab("Timing Error")

barpl7 <- ggplot(finalDF[finalDF$variable == "rmse",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3)), size = 7) + 
  theme_gray(base_size = 14) +
  xlab("Model") + ylab("Mean RMSE")

barpl8 <- ggplot(finalDF[finalDF$variable == "corr",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3)), size = 7) + 
  theme_gray(base_size = 14) + 
  xlab("Model") + ylab("Mean Corr. Coefficient")

