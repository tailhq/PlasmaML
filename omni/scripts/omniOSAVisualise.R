library(plyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(latex2exp)

setwd("Development/PlasmaML/data")

df <- read.csv("OmniOSARes.csv", 
               header = TRUE, 
               stringsAsFactors = FALSE)

ggplot(df) + 
  geom_bar(stat="identity",aes(order, mae)) + 
  facet_grid(data~model~globalOpt) +
  theme_gray(base_size = 14)