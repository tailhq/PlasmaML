#! /usr/bin/Rscript
library(plyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(latex2exp)
setwd("../../../DynaML/data/")

dfPredNAR <- read.csv("PredOmniARStormsRes.csv", 
                      header = FALSE, col.names = c("Dst", "NAR"))

dfPredNARX <- read.csv("PredOmniARXStormsRes.csv", 
                       header = FALSE, col.names = c("Dst", "NARX"))

dfPredTL <- read.csv("OmniTLPredStormsRes.csv", 
                     header = FALSE, col.names = c("Dst", "TL"))

dfPredNM <- read.csv("OmniNMPredStormsRes.csv",
                     header = FALSE, 
                     col.names = c("Dst", "NM"), 
                     na.strings = c("NAN"))

cumDF <- data.frame(absErr = c(abs(dfPredNAR$Dst - dfPredNAR$NAR), 
                               abs(dfPredNARX$Dst - dfPredNARX$NAR), 
                               abs(dfPredTL$Dst - dfPredTL$TL), 
                               abs(dfPredNM$Dst - dfPredNM$NM)), 
                    model = c(rep("GP-AR", nrow(dfPredNAR)), 
                              rep("GP-ARX", nrow(dfPredNARX)), 
                              rep("TL", nrow(dfPredTL)), 
                              rep("NM", nrow(dfPredNM))))

cumDFRel <- data.frame(absErr = c(abs((dfPredNAR$Dst - dfPredNAR$NAR)/abs(dfPredNAR$Dst)), 
                               abs((dfPredNARX$Dst - dfPredNARX$NAR)/abs(dfPredNARX$Dst)), 
                               abs((dfPredTL$Dst - dfPredTL$TL)/abs(dfPredTL$Dst)), 
                               abs((dfPredNM$Dst - dfPredNM$NM)/abs(dfPredNM$Dst))), 
                    model = c(rep("GP-AR", nrow(dfPredNAR)), 
                              rep("GP-ARX", nrow(dfPredNARX)), 
                              rep("TL", nrow(dfPredTL)), 
                              rep("NM", nrow(dfPredNM))))

ggplot(cumDF, aes(absErr, colour = model)) + stat_ecdf() +
  theme_gray(base_size = 22) +
  xlab(TeX('$|Dst - \\hat{D}st|$')) + ylab("Cumulative Probability") + 
  scale_x_continuous(breaks = round(seq(min(cumDF$absErr, na.rm = TRUE), 
                                        max(cumDF$absErr, na.rm = TRUE), 
                                        by = 5),1)) +
  scale_y_continuous(breaks = round(seq(0, 1.0, by = 0.1), 2)) + 
  coord_cartesian(xlim = c(0, 80))

ggplot(cumDFRel, aes(absErr, colour = model)) + stat_ecdf(size = 1.25) +
  theme_gray(base_size = 22)  + 
  scale_x_continuous(breaks = round(seq(0.0, 
                                        1.0, 
                                        by = 0.05),1)) +
  scale_y_continuous(breaks = round(seq(0, 1.0, by = 0.1), 2)) + 
  xlab(TeX('$|Dst - \\hat{D}st|/|Dst|$')) + ylab("Cumulative Probability") + 
  coord_cartesian(xlim = c(0, 1.0))

tlP <- read.csv("TLFinalPredRes.csv", header = FALSE, col.names = c("actual", "Dst"))
arxP <- read.csv("ARXFinalPredRes.csv", header = FALSE, col.names = c("actual", "Dst"))

nmP <- read.csv("NMFinalPredRes.csv", header = FALSE, col.names = c("actual", "Dst"))


arx_errorbars_pred <- read.csv("ARXErrorBarsFinalPredRes.csv", 
                               header = FALSE, 
                               col.names = c("Dst", "predicted", "lower", "upper"))
arx_errorbars_pred$time <- 1:nrow(arx_errorbars_pred)

palette1 <- c("#000000", "firebrick3", "forestgreen", "steelblue2")
lines1 <- c("solid", "solid", "dotdash", "dotdash")

meltedPred <- melt(arx_errorbars_pred, id="time")

ggplot(meltedPred, aes(x=time,y=value, colour=variable, linetype=variable)) +
  geom_line(size=1.35) +
  theme_gray(base_size = 22) + 
  scale_colour_manual(values=palette1) + 
  scale_linetype_manual(values = lines1, guide=FALSE) +
  xlab("Time (hours)") + ylab("Dst (nT)")

ar_errorbars_pred <- read.csv("ARErrorBarsFinalPredRes.csv", 
                              header = FALSE, 
                              col.names = c("Dst", "predicted", 
                                            "lower", "upper"))
ar_errorbars_pred$time <- 1:nrow(ar_errorbars_pred)

meltedPred1 <- melt(ar_errorbars_pred, id="time")

ggplot(meltedPred1, aes(x=time,y=value, colour=variable, linetype=variable)) +
  geom_line(size=1.35) +
  theme_gray(base_size = 22) + 
  scale_colour_manual(values=palette1) + 
  scale_linetype_manual(values = lines1, guide=FALSE) +
  xlab("Time (hours)") + ylab("Dst (nT)")

dstDF <- data.frame(cbind(tlP$actual, 1:nrow(tlP)))
colnames(dstDF) <- c("Dst", "timestamp")
dstDF$model <- rep("Dst", nrow(tlP))

tlP$actual <- NULL
tlP$timestamp <- 1:nrow(tlP)
tlP$model <- rep("TL", nrow(tlP))

nmP$actual <- NULL
nmP$timestamp <- 1:nrow(nmP)
nmP$model <- rep("NM", nrow(nmP))

arxP$actual <- NULL
arxP$timestamp <- 1:nrow(arxP)
arxP$model <- rep("GP-ARX", nrow(arxP))

dfP <- rbind(dstDF, tlP, nmP, arxP)

colnames(dfP) <- c("Dst", "time", "model")

cbPalette <- c("#000000", "firebrick3", "olivedrab3", "steelblue3")
lt <- c("Dst"="solid", "GP-ARX"="solid", "NM"="4C88C488", "TL"="12345678")
l <- c("4C88C488", "12345678", "solid", "solid")

dfP$ltype <- with(dfP, lt[model])

ggplot(dfP, aes(x = time, y = Dst, linetype=factor(ltype), legend=FALSE)) +
  geom_line(aes(colour=model), size=2) + 
  theme_gray(base_size = 22) + 
  scale_colour_manual(values=cbPalette) + 
  scale_linetype_manual(values = l, guide=FALSE) +
  xlab("Time (hours)") + ylab("Dst (nT)")

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

dfNM <- read.csv("OmniNMStormsRes.csv", 
                  header = FALSE, stringsAsFactors = TRUE, 
                  col.names = c("eventID","stormCat","order", "modelSize",
                                "rmse", "corr", "deltaDstMin", "DstMin",
                                "deltaT"), na.strings = c("NaN"))

dfTL <- read.csv("OmniTLStormsRes.csv", 
                 header = FALSE, stringsAsFactors = TRUE, 
                 col.names = c("eventID","stormCat","order", "modelSize",
                               "rmse", "corr", "deltaDstMin", "DstMin",
                               "deltaT"))

dfNM$model <- rep("NM(CWI)", nrow(dfNM))

dfTL$model <- rep("TL(CWI)", nrow(dfTL))

dfPer$model <- rep("Persist(1)", nrow(dfPer))

df2$model <- rep("GP-AR", nrow(df2))

df$model <- rep("GP-ARX1", nrow(df))

dfVBz$model <-rep("GP-ARX", nrow(dfVBz))


bindDF <- rbind(df, df2, dfPer, dfVBz, dfNM, dfTL)


meltedDF <- melt(bindDF[bindDF$order == 6 | bindDF$model != "GP-AR1",],
                 id.vars=c("model", "stormCat", "eventID"), na.rm = TRUE)

meltedDF1 <- meltedDF[meltedDF$variable != "order" & 
                        meltedDF$variable != "modelSize" &
                        meltedDF$variable != "DstMin",]

meansGlobal <- ddply(meltedDF1, c("model", "variable"), summarise,
               meanValue=mean(value))

meansGlobalAbs <- ddply(meltedDF1, c("model", "variable"), summarise,
                     meanValue=mean(abs(value)))

dfother <- read.csv("resultsModels.csv", 
                header = FALSE, stringsAsFactors = TRUE, 
                col.names = c("model","variable","meanValue"))

finalDF <- rbind(dfother, 
                 meansGlobal[!(meansGlobalAbs$variable %in% c("deltaT", "deltaDstMin")) & 
                                        meansGlobal$model %in% c("TL(CWI)","NM(CWI)","GP-AR", "Persist(1)",
                                                                 "GP-ARX", "GP-ARX1"),], 
                 meansGlobalAbs[meansGlobalAbs$variable %in% c("deltaT", "deltaDstMin") & 
                                  meansGlobalAbs$model %in% c("TL(CWI)","NM(CWI)","GP-AR", "Persist(1)",
                                                              "GP-ARX", "GP-ARX1"),])

finalDF$colorVal <- with(finalDF, ifelse(!(model %in% c("Persist(1)", "GP-AR", "GP-ARX")), 0, 
                                     match(model, c("Persist(1)", "GP-AR", "GP-ARX"))))


barplrmse1 <- ggplot(dfother[dfother$variable == "rmse",], 
                 aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 22) +
  xlab("Model") + ylab("Mean RMSE")

colourPalette <- c("0" = "grey38", "1" = "steelblue3", 
                   "2" = "firebrick2", "3" = "firebrick3")

barplrmse2 <- ggplot(finalDF[finalDF$variable == "rmse" & 
                               !(finalDF$model %in% c("GP-ARX1", "GP-AR", "GP-ARX")),], 
                     aes(x = reorder(model, desc(meanValue)), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  theme_gray(base_size = 20) + 
  xlab("Model") + ylab("Mean RMSE")


barplrmse3 <- ggplot(finalDF[finalDF$variable == "rmse" & !(finalDF$model %in% c("GP-ARX1")),], 
                     aes(x = reorder(model, desc(meanValue)), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 22) +
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  xlab("Model") + ylab("Mean RMSE")



barplcc1 <- ggplot(dfother[dfother$variable == "corr",], 
                     aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 22) +
  xlab("Model") + ylab("Mean RMSE")


barplcc2 <- ggplot(finalDF[finalDF$variable == "corr" & 
                               !(finalDF$model %in% c("GP-ARX1", "GP-AR", "GP-ARX")),], 
                     aes(x = reorder(model, meanValue), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 2)), size = 7, nudge_y = 1.25) + 
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  theme_gray(base_size = 20) + 
  xlab("Model") + ylab("Mean RMSE")


barplcc3 <- ggplot(finalDF[finalDF$variable == "corr" & !(finalDF$model %in% c("GP-ARX1")),], 
                     aes(x = reorder(model, meanValue), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 2)), size = 7, nudge_y = 0.05) + 
  theme_gray(base_size = 22) +
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  xlab("Model") + ylab("Mean Corr. Coefficient")




barpl5 <- ggplot(finalDF[finalDF$variable == "deltaDstMin" & !(finalDF$model %in% c("GP-ARX1")),], 
                 aes(x = reorder(model, desc(meanValue)), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) +
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  theme_gray(base_size = 22) +
  xlab("Model") + ylab(TeX('$\\bar{\\Delta Dst_{min}}$'))

barpl6 <- ggplot(finalDF[finalDF$variable == "deltaT" & !(finalDF$model %in% c("GP-ARX1")),], 
                 aes(x=reorder(model, desc(meanValue)), y=meanValue, fill=factor(colorVal))) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 0.1) + 
  scale_fill_manual(values = colourPalette, guide=FALSE) +
  theme_gray(base_size = 22) +
  xlab("Model") + ylab(TeX('$\\bar{| \\Delta t_{peak} |$'))

barpl7 <- ggplot(finalDF[finalDF$variable == "rmse",], 
                 aes(x = reorder(model, desc(meanValue)), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 1)), size = 7, nudge_y = 1.25) + 
  theme_gray(base_size = 22) +
  xlab("Model") + ylab("Mean RMSE")

barpl8 <- ggplot(finalDF[finalDF$variable == "corr",], aes(x = reorder(model, meanValue), y=meanValue)) + 
  geom_bar(stat="identity", position="dodge") + 
  geom_text(aes(label = round(meanValue, digits = 2)), size = 7, nudge_y = 0.025) + 
  theme_gray(base_size = 22) + 
  xlab("Model") + ylab("Mean Corr. Coefficient")

deltaDstPlot <- ggplot(dfVBz, aes(x=DstMin, y=deltaDstMin/DstMin)) + 
  geom_point(aes(color=as.factor(stormCat))) + theme_gray(base_size = 22) + 
  labs(x = TeX('$min(D_{st})$'), 
       y=TeX('$\\frac{\\Delta D_{st}}{min(D_{st})}$'), color="Storm Category")

library(directlabels)
setwd("../../PlasmaMLSpark/data")
lDF <- read.csv("OmniARXLandscapeRes.csv", col.names=c("rmse", "degree", "b", "sigma"))
contDF <- lDF[lDF$b <= 0.1, c("rmse", "b", "sigma")]

c <- ggplot(contDF, aes(x = b, y = sigma, z=rmse)) + 
  stat_contour(aes(colour = ..level..), binwidth  = 0.02, size = 0.75) + 
  theme_gray(base_size = 22) + 
  scale_x_continuous(breaks = round(seq(0.0, 0.1, by = 0.02),2)) + #scale_y_log10() +
  xlab("b") + ylab(TeX('$\\sigma$'))
direct.label(c, "top.points")

lDF1 <- read.csv("OmniARXLandscapeMLRes.csv", col.names=c("rmse", "degree", "b", "sigma"))
contDF1 <- lDF1[, c("rmse", "b", "sigma")]

ggplot(contDF1, aes(x = b, y = sigma, z=rmse)) + 
  stat_contour(aes(colour = ..level..), binwidth  = 10.0, size = 1.0)

lDF2 <- read.csv("OmniARXLandscapeFBMRes.csv", col.names=c("rmse", "hurst", "sigma"))

ggplot(lDF2, aes(x = hurst, y = sigma, z=rmse)) + stat_contour(aes(colour = ..level..), binwidth  = 0.02, size = 1.0)