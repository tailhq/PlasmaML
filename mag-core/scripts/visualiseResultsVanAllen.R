# Title     : TODO
# Objective : TODO
# Created by: mandar
# Created on: 18/09/2018

#/usr/bin/Rscript
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
lossFlag <- args[2]
setwd(direc)

posterior_samples <- read.csv("posterior_samples.csv", header = TRUE)
prior_samples <- read.csv("prior_samples.csv", header = TRUE)


if(lossFlag == "loss") {
    qplot(
    prior_samples$"lambda_beta",  prior_samples$"lambda_b",
    xlab = expression(beta), ylab=expression('b'))

    ggsave("scatter_beta_b_prior.png")

    qplot(
    posterior_samples$"lambda_beta", posterior_samples$"lambda_b",
    xlab = expression(beta), ylab=expression('b'))
    ggsave("scatter_beta_b_posterior.png")

    qplot(
    prior_samples$"lambda_beta",  exp(prior_samples$"lambda_alpha"),
    xlab = expression(beta), ylab=expression(alpha))
    ggsave("scatter_beta_alpha_prior.png")

    qplot(
    posterior_samples$"lambda_beta", exp(posterior_samples$"lambda_alpha"),
    xlab = expression(beta), ylab=expression(alpha))
    ggsave("scatter_beta_alpha_posterior.png")

    ggplot(prior_samples, aes(x=lambda_beta)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        theme_gray(base_size = 24) +
        xlab(expression(beta))
    ggsave("histogram_beta_prior.png")

    ggplot(posterior_samples, aes(x=lambda_beta)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        theme_gray(base_size = 24) +
        xlab(expression(beta))

    ggsave("histogram_beta_posterior.png")


    ggplot(prior_samples, aes(x=lambda_b)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        theme_gray(base_size = 24) +
        xlab("b")

    ggsave("histogram_b_prior.png")

    ggplot(posterior_samples, aes(x=lambda_b)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        theme_gray(base_size = 24) +
        xlab("b")

    ggsave("histogram_b_posterior.png")

    ggplot(prior_samples, aes(x=exp(lambda_alpha))) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        theme_gray(base_size = 24) +
        xlab(expression(alpha))

    ggsave("histogram_alpha_prior.png")

    ggplot(posterior_samples, aes(x=exp(lambda_alpha))) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        theme_gray(base_size = 24) +
        xlab(expression(alpha))

    ggsave("histogram_alpha_posterior.png")

} else {

    qplot(
    prior_samples$"Q_gamma",  prior_samples$"Q_b",
    xlab = expression(gamma), ylab = expression('b'))
    ggsave("scatter_gamma_b_prior.png")

    qplot(
    posterior_samples$"Q_gamma", posterior_samples$"Q_b",
    xlab = expression(gamma), ylab = expression('b'))
    ggsave("scatter_gamma_b_posterior.png")

    ggplot(prior_samples[prior_samples$Q_gamma < 100,], aes(x=Q_gamma)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab(expression(gamma))

    ggsave("histogram_gamma_prior.png")

    ggplot(posterior_samples, aes(x=Q_gamma)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab(expression(gamma))

    ggsave("histogram_gamma_posterior.png")


    ggplot(prior_samples, aes(x=Q_b)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab("b")

    ggsave("histogram_b_prior.png")


    ggplot(posterior_samples, aes(x=Q_b)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab("b")

    ggsave("histogram_b_posterior.png")

    ggplot(prior_samples, aes(x=Q_beta)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab(expression(beta))

    ggsave("histogram_beta_prior.png")


    ggplot(posterior_samples, aes(x=Q_beta)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab(expression(beta))

    ggsave("histogram_beta_posterior.png")

    ggplot(prior_samples, aes(x=Q_alpha)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab(expression(alpha))

    ggsave("histogram_alpha_prior.png")


    ggplot(posterior_samples, aes(x=Q_alpha)) +
        geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
        binwidth=.5,
        colour="black", fill="white") +
        geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
        xlab(expression((alpha)))

    ggsave("histogram_alpha_posterior.png")

}


