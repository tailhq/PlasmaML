#/usr/bin/Rscript
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
lossFlag <- args[2]
setwd(direc)

posterior_samples <- read.csv("posterior_samples.csv", header = TRUE)
prior_samples <- read.csv("prior_samples.csv", header = TRUE)

ground_truth <- read.csv("diffusion_params.csv", header = TRUE, na.strings = c("-Infinity"))


if(lossFlag == "loss") {
  qplot(
    prior_samples$"tau_beta",  prior_samples$"tau_b",
    xlab = expression(beta), ylab=expression(b))

  ggsave("scatter_beta_b_prior.png")

  qplot(
    posterior_samples$"tau_beta", posterior_samples$"tau_b",
    xlab = expression(beta), ylab=expression(b))
  ggsave("scatter_beta_b_posterior.png")

  qplot(
    prior_samples$"tau_beta",  exp(prior_samples$"tau_alpha"),
    xlab = expression(beta), ylab=expression(alpha))
  ggsave("scatter_beta_alpha_prior.png")

  qplot(
    posterior_samples$"tau_beta", exp(posterior_samples$"tau_alpha"),
    xlab = expression(beta), ylab=expression(alpha))
  ggsave("scatter_beta_alpha_posterior.png")

  ggplot(prior_samples, aes(x=tau_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("beta") +
    geom_vline(aes(xintercept=ground_truth$tau_beta),   # Ignore NA values for mean
               color="red", linetype="dashed", size=.5)
  ggsave("histogram_beta_prior.png")

  ggplot(posterior_samples, aes(x=tau_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("beta") +
    geom_vline(aes(xintercept=ground_truth$tau_beta),   # Ignore NA values for mean
               color="red", linetype="dashed", size=.5)
  ggsave("histogram_beta_posterior.png")


  ggplot(prior_samples, aes(x=tau_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$tau_b),   # Ignore NA values for mean
               color="red", linetype="dashed", size=0.5)
  ggsave("histogram_b_prior.png")

  ggplot(posterior_samples, aes(x=tau_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$tau_b),   # Ignore NA values for mean
               color="red", linetype="dashed", size=0.5)
  ggsave("histogram_b_posterior.png")

  ggplot(prior_samples, aes(x=exp(tau_alpha))) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("alpha") +
    geom_vline(aes(xintercept=exp(ground_truth$tau_alpha)),   # Ignore NA values for mean
               color="red", linetype="dashed", size=0.5)
  ggsave("histogram_alpha_prior.png")

  ggplot(posterior_samples, aes(x=exp(tau_alpha))) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("alpha") +
    geom_vline(aes(xintercept=exp(ground_truth$tau_alpha)),   # Ignore NA values for mean
               color="red", linetype="dashed", size=0.5)
  ggsave("histogram_alpha_posterior.png")

} else {

  qplot(
    prior_samples$"Q_gamma",  prior_samples$"Q_b",
    xlab = expression(gamma), ylab = expression(b))
  ggsave("scatter_gamma_b_prior.png")

  qplot(
    posterior_samples$"Q_gamma", posterior_samples$"Q_b",
    xlab = expression(gamma), ylab = expression(b))
  ggsave("scatter_gamma_b_posterior.png")

  ggplot(prior_samples[prior_samples$Q_gamma < 100,], aes(x=Q_gamma)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab(expression(gamma)) +
    geom_vline(aes(xintercept=ground_truth$Q_gamma),   # Ignore NA values for mean
               color="red", linetype="dashed", size=.5)
  ggsave("histogram_gamma_prior.png")

  ggplot(posterior_samples, aes(x=Q_gamma)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab(expression(gamma)) +
    geom_vline(aes(xintercept=ground_truth$Q_gamma),   # Ignore NA values for mean
               color="red", linetype="dashed", size=.5)
  ggsave("histogram_gamma_posterior.png")


  ggplot(prior_samples, aes(x=Q_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="red", linetype="dashed", size=0.5)
  ggsave("histogram_b_prior.png")


  ggplot(posterior_samples, aes(x=Q_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="red", linetype="dashed", size=0.5)
  ggsave("histogram_b_posterior.png")

}

