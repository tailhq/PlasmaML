#/usr/bin/Rscript
library(ggplot2)
library(reshape2)
library(latex2exp)
library(directlabels)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
file  <- args[2]

setwd(direc)

scatter_df <- read.csv(file, header = FALSE)
colnames(scatter_df) <- c("Prediction", "Actual", "TimeLag")


ggplot(scatter_df, aes(x=Actual, y=Prediction)) +
  theme_gray(base_size = 20) + 
  geom_point(alpha=1/3) +    
  geom_smooth(method=lm) +
  ylab(TeX('$\\hat{v}$  (km/s)')) +
  xlab(TeX('$v$  (km/s)'))
                             


ggplot(scatter_df, aes(x=Prediction, y=TimeLag)) +
  theme_gray(base_size = 20) + 
  geom_point(alpha=1/3) +
  xlab(TeX('$\\hat{v}$  (km/s)')) +
  ylab(TeX('$arg\\,max_{i} \ \ \\left(\\hat{p}_{i}\\right)$  (hr)'))