library(plyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(latex2exp)

palette1 <- c("#000000", "firebrick3", "forestgreen", "steelblue2")
lines1 <- c("solid", "solid", "dotdash", "dotdash")
setwd("Development/PlasmaML/data/")


# Compare Brier score of naive models vs GP-ARX
df <- read.csv("brier_scores.csv", 
               header = FALSE, 
               stringsAsFactors = FALSE, 
               colClasses = rep("numeric", 2), 
               col.names = c("prob","brier"))


ggplot(df, aes(x=prob,y=brier)) +
  geom_path(size=1.25) + geom_hline(aes(yintercept=0.076, linetype = "dotdash"), show.legend = TRUE) +
  theme_gray(base_size = 22) + 
  scale_colour_manual(values=palette1) + 
  scale_linetype_manual(values = lines1, guide=FALSE) +
  xlab("Probability of Storm Jump") + ylab("Brier Score")


arx_errorbars_pred <- read.csv("mogp_preds_errorbars2.csv", 
                               header = FALSE, 
                               col.names = c("Dst", "predicted", "lower", "upper"))
arx_errorbars_pred$time <- 1:nrow(arx_errorbars_pred)

meltedPred <- melt(arx_errorbars_pred, id="time")

ggplot(meltedPred, aes(x=time,y=value, colour=variable, linetype=variable)) +
  geom_line(size=1.35) +
  theme_gray(base_size = 22) + 
  scale_colour_manual(values=palette1) + 
  scale_linetype_manual(values = lines1, guide=FALSE) +
  xlab("Time (hours)") + ylab("Dst (nT)")

arx_onset_pred <- read.csv("mogp_onset_predictions1.csv", 
                               header = FALSE, 
                               col.names = c("p", "storm"))
arx_onset_pred$time <- 1:nrow(arx_onset_pred)

meltedPred1 <- melt(arx_onset_pred, id="time")

ggplot(meltedPred1, aes(x=time,y=value, colour=variable, linetype=variable)) +
  geom_line(size=1.35) +
  theme_gray(base_size = 22) + 
  scale_colour_manual(values=palette1) + 
  scale_linetype_manual(values = lines1, guide=FALSE) +
  xlab("Time (hours)") + ylab("P(Dst(t+4) - Dst(t) < -70 nT)")
