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
colnames(scatter_df) <- c("Prediction", "Actual", "TimeLag")

palette <- c("#000000", "#CC0C0099", "#5C88DA99")

ggplot(scatter_df, aes(x=Actual, y=Prediction)) +
  theme_gray(base_size = 20) + 
  geom_point(alpha=0.3) + 
  stat_density_2d(colour = "blue", size=0.65, alpha = 0.85) +
  geom_abline(slope = 1, intercept = 0, color = "#CC0C0099", size=1.1, alpha=1) +
  ylab(TeX('$\\hat{v}$  (km/s)')) +
  xlab(TeX('$v$  (km/s)'))


ggsave(paste(iden, "scatter_v.pdf", sep = ''), scale = 1.0, device = pdf())
                             


ggplot(scatter_df, aes(x=Prediction, y=TimeLag)) +
  theme_gray(base_size = 20) + 
  geom_point(alpha=1/3) +
  xlab(TeX('$\\hat{v}$  (km/s)')) +
  ylab(TeX('$arg\\,max_{i} \ \ \\left(\\hat{p}_{i}\\right)$  (hr)'))

ggsave(paste(iden, "scatter_v_tl.pdf", sep = ''), scale = 1.0, device = pdf())