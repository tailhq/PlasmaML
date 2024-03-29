#/usr/bin/Rscript
library(ggplot2)
library(latex2exp)

args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
lossFlag <- args[2]
setwd(direc)

palette2 <- c("firebrick3", "#000000")
palette3 <- c("#CC0C0099", "#5C88DA99")

psd_df <- read.csv("solution.csv")
s_df <- read.csv("sensitivity.csv")
s_df <- subset(s_df, parameter != "gamma")

s_df_q <- subset(s_df, quantity == "Q")
s_df_b <- subset(s_df, parameter == "b")
s_df_rest <- subset(s_df, parameter != "b" & quantity != "Q")

s_df$parameter <- factor(as.character(s_df$parameter), labels = c("alpha", "b", "beta"))
s_df_q$parameter <- factor(as.character(s_df_q$parameter), labels = c("alpha", "b", "beta"))
s_df_rest$parameter <- factor(as.character(s_df_rest$parameter), labels = c("alpha", "beta"))


s_df$quantity <- factor(
    s_df$quantity, 
    levels = c("Q", "dll", "lambda"),
    labels = unname(TeX(c('$q(L^*, t)$', '$\\kappa(L^*, t)$', '$\\lambda(L^*, t)$'))))

s_df_b$quantity <- factor(
  s_df_b$quantity, 
  levels = c("Q", "dll", "lambda"),
  labels = unname(TeX(c('$q(L^*, t)$', '$\\kappa(L^*, t)$', '$\\lambda(L^*, t)$'))))

s_df_rest$quantity <- factor(
  s_df_rest$quantity, 
  levels = c("dll", "lambda"),
  labels = unname(TeX(c('$\\kappa(L^*, t)$', '$\\lambda(L^*, t)$')))
)

#c("kappa(L^*, t)", "lambda(L^*, t)", "Q(L^*, t)")

ggplot(psd_df, aes(x = t, y = l)) +
    geom_raster(aes(fill = psd)) +
    scale_fill_viridis_c(name = unname(TeX('$f(L^{*}, t)$'))) +
    theme_gray(base_size = 20) +
    ylab(TeX('$L^{*}$')) +
    xlab(TeX('$t$')) +
    theme(
    #legend.position="top", 
    #legend.direction="vertical")#, 
    legend.text=element_text(size=14))

ggsave("psd.pdf", scale=1.5)



ggplot(s_df, aes(x = t, y = l)) +
    geom_raster(aes(fill = value)) +
    facet_grid(quantity ~ parameter, labeller = label_parsed) +
    scale_fill_viridis_c(name = "") +
    theme_gray(base_size = 20) +
    ylab(TeX('$L^{*}$')) +
    xlab(TeX('$t$')) +
    theme(axis.text.x = element_text(angle = 90), legend.text=element_text(size=14))

ggsave("sensitivity.pdf", scale = 2.0)

ggplot(s_df_q, aes(x = t, y = l)) +
    geom_raster(aes(fill = value)) +
    facet_grid(parameter~., labeller = label_parsed) +
    scale_fill_viridis_c(name = TeX('$s(L^{*}, t)$')) +
    theme_gray(base_size = 20) +
    ylab(TeX('$L^{*}$')) +
    xlab(TeX('$t$')) +
    theme(
    #legend.position="top", 
    #legend.direction="vertical")#, 
    legend.text=element_text(size=14))

ggsave("sensitivity_Q.pdf", scale=1.5)

ggplot(s_df_b, aes(x = t, y = l)) +
  geom_raster(aes(fill = value)) +
  facet_grid(quantity~., labeller = label_parsed) +
  scale_fill_viridis_c(name = TeX('$s(L^{*}, t)$')) +
  theme_gray(base_size = 20) +
  ylab(TeX('$L^{*}$')) +
  xlab(TeX('$t$')) +
  theme(
    #legend.position="top", 
    #legend.direction="vertical")#, 
    legend.text=element_text(size=14))

ggsave("sensitivity_b.pdf", scale=1.5)

ggplot(s_df_rest, aes(x = t, y = l)) +
  geom_raster(aes(fill = value)) +
  facet_grid(quantity ~ parameter, labeller = label_parsed) +
  scale_fill_viridis_c(name = TeX('$s(L^{*}, t)$')) +
  theme_gray(base_size = 20) +
  ylab(TeX('$L^{*}$')) +
  xlab(TeX('$t$')) +
  theme(
    #legend.position="top", 
    #legend.direction="vertical")#, 
    legend.text=element_text(size=14))

ggsave("sensitivity_rest.pdf", scale=1.5)