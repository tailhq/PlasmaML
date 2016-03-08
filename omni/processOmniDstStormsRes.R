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


p8 <- qplot(data = df, x = factor(order), y = rmse,
            facets = .~stormCat, geom = "boxplot", 
            xlab = "Model Order", ylab = "Root Mean Sq. Error")

p9 <- qplot(data = df, x = factor(order), y = corr,
            facets = .~stormCat, geom = "boxplot", 
            xlab = "Model Order", ylab = "Cross Correlation")

p10 <- qplot(data = df, x = factor(order), y = deltaDstMin,
             facets = .~stormCat, geom = "boxplot", 
             xlab = "Model Size", ylab = "Delta Dst Min")

p11 <- qplot(data = df, x = factor(order), y = deltaT,
             facets = .~stormCat, geom = "boxplot", 
             xlab = "Model Order", ylab = "Timing Error")


qplot(data = df, x = factor(order), y = rmse,
      geom = "boxplot", 
      xlab = "Model Order", ylab = "Root Mean Sq. Error")

qplot(data = df, x = factor(order), y = corr,
      geom = "boxplot", 
      xlab = "Model Order", ylab = "Cross Correlation")


df1 <- read.csv("OmniARStormsRes.csv", 
               header = FALSE, stringsAsFactors = TRUE, 
               col.names = c("eventID","stormCat","order", "modelSize",
                             "rmse", "corr", "deltaDstMin", "DstMin",
                             "deltaT"))


p12 <- qplot(data = df1, x = factor(order), y = rmse,
            facets = .~stormCat, geom = "boxplot", 
            xlab = "Model Order", ylab = "Root Mean Sq. Error")

p13 <- qplot(data = df1, x = factor(order), y = corr,
            facets = .~stormCat, geom = "boxplot", 
            xlab = "Model Order", ylab = "Cross Correlation")

p14 <- qplot(data = df1, x = factor(order), y = deltaDstMin,
             facets = .~stormCat, geom = "boxplot", 
             xlab = "Model Size", ylab = "Delta Dst Min")


p15 <- qplot(data = df1, x = factor(order), y = deltaT,
             facets = .~stormCat, geom = "boxplot", 
             xlab = "Model Order", ylab = "Timing Error")


qplot(data = df1, x = factor(order), y = rmse,
      geom = "boxplot", 
      xlab = "Model Order", ylab = "Root Mean Sq. Error")

qplot(data = df1, x = factor(order), y = corr,
      geom = "boxplot", 
      xlab = "Model Order", 
      ylab = "Cross Correlation")

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

df1$model <- rep("NAR-FBM", nrow(df1))

dfVBz$model <-rep("NARX-Poly-VBz", nrow(dfVBz))

dfPVBz$model <- rep("NARX-Poly-PVBz", nrow(dfPVBz))

bindDF <- rbind(df, df1, df2, dfPer, dfVBz, dfPVBz)



qplot(data = bindDF, x = factor(order), y = rmse,
      geom = "boxplot", fill = model, facets = .~stormCat,
      xlab = "Model Order", ylab = "Root Mean Sq. Error")

qplot(data = bindDF, x = factor(order), y = corr,
      geom = "boxplot", fill = model, facets = .~stormCat,
      xlab = "Model Order", ylab = "Cross Correlation")

qplot(data = bindDF, x = factor(order), y = deltaDstMin,
      geom = "boxplot", fill = model, facets = .~stormCat,
      xlab = "Model Order", ylab = "Delta Dst Min")



qplot(data = bindDF, x = factor(order), y = deltaDstMin,
      geom = "boxplot", fill = model,
      xlab = "Model Order", ylab = "Delta Dst Min")

qplot(data = bindDF, x = factor(order), y = rmse,
      geom = "boxplot", fill = model,
      xlab = "Model Order", ylab = "Root Mean Sq. Error")


qplot(data = bindDF, x = factor(order), y = corr,
      geom = "boxplot", fill = model, 
      xlab = "Model Order", ylab = "Cross Correlation")

qplot(data = bindDF, x = factor(order), y = deltaT,
      geom = "boxplot", fill = model, 
      xlab = "Model Order", ylab = "Timing Error")



df3 <- read.csv("PolyDstPVOmniARXStormsRes.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("eventID","stormCat","order", "modelSize",
                              "rmse", "corr", "deltaDstMin", "DstMin",
                              "deltaT"))

p8 <- qplot(data = df3, x = factor(order), y = rmse,
            facets = .~stormCat, geom = "boxplot", 
            xlab = "Model Order", ylab = "Root Mean Sq. Error")

p9 <- qplot(data = df3, x = factor(order), y = corr,
            facets = .~stormCat, geom = "boxplot", 
            xlab = "Model Order", ylab = "Cross Correlation")

p10 <- qplot(data = df3, x = factor(order), y = deltaDstMin,
             facets = .~stormCat, geom = "boxplot", 
             xlab = "Model Size", ylab = "Delta Dst Min")

p11 <- qplot(data = df3, x = factor(order), y = deltaT,
             facets = .~stormCat, geom = "boxplot", 
             xlab = "Model Order", ylab = "Timing Error")


qplot(data = bindDF[bindDF$order == 1 & bindDF$model != "NAR-FBM",], x = deltaDstMin,
      geom = "histogram", facets = model~.,
      xlab = "delta Dst min", ylab = "Frequency")


meltedDF <- melt(bindDF[(bindDF$order == 1 | bindDF$model == "Persist(1)") & bindDF$model != "NAR-FBM",],
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
  geom_text(aes(label = round(meanValue, digits = 3))) + 
  xlab("Model") + ylab("delta(Dst min)")

barpl6 <- ggplot(finalDF[finalDF$variable == "deltaT",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3))) + 
  xlab("Model") + ylab("Timing Error")

barpl7 <- ggplot(finalDF[finalDF$variable == "rmse",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3))) + 
  xlab("Model") + ylab("Mean RMSE")

barpl8 <- ggplot(finalDF[finalDF$variable == "corr",], aes(x=model, y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 3))) + 
  xlab("Model") + ylab("Mean Corr. Coefficient")

