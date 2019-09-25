#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
library(latex2exp)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
file  <- args[2]
iden  <- args[3]

setwd(direc)

scatter_df <- read.csv(file, header = FALSE)
colnames(scatter_df) <- c("time", "Value", "Type")


palette <- c("#000000", "#CC0C0099", "#5C88DA99")
palette1 <- c("firebrick3", "gray27", "forestgreen")


ggplot(scatter_df, aes(x=time, y=Value, color=Type)) + 
  geom_point(size = 0.75) + 
  geom_line() +
  scale_colour_manual(labels = c("Prediction", "Actual"), values=palette1) +
  theme_gray(base_size = 20) +
  theme(legend.title=element_blank(), legend.position = "top") +
  ylab("km/s") +
  xlab(TeX('$t$  (hours)'))

ggsave(paste(iden, "ts.pdf", sep = ''), scale = 1.0, device = pdf())