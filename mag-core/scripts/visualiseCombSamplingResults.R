#/usr/bin/Rscript
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
setwd(direc)

posterior_samples <- read.csv("posterior_samples.csv", header = TRUE)
prior_samples <- read.csv("prior_samples.csv", header = TRUE)

ground_truth <- read.csv("diffusion_params.csv", header = TRUE, na.strings = c("-Infinity"))

qplot(
  prior_samples$"lambda_beta",  prior_samples$"lambda_b",
  xlab = expression(beta), ylab=expression(b))

ggsave("scatter_lambda_beta_b_prior.png")

qplot(
  posterior_samples$"lambda_beta", posterior_samples$"lambda_b",
  xlab = expression(beta), ylab=expression(b))
ggsave("scatter_lambda_beta_b_posterior.png")

qplot(
  prior_samples$"lambda_beta",  exp(prior_samples$"lambda_alpha"),
  xlab = expression(beta), ylab=expression(alpha))
ggsave("scatter_lambda_beta_alpha_prior.png")

qplot(
  posterior_samples$"lambda_beta", exp(posterior_samples$"lambda_alpha"),
  xlab = expression(beta), ylab=expression(alpha))
ggsave("scatter_lambda_beta_alpha_posterior.png")

ggplot(prior_samples, aes(x=lambda_beta)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(beta)) +
  geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
             color="red", linetype="dashed", size=.75)
ggsave("histogram_lambda_beta_prior.png")

ggplot(posterior_samples, aes(x=lambda_beta)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(beta)) +
  geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
             color="red", linetype="dashed", size=.75)
ggsave("histogram_lambda_beta_posterior.png")


ggplot(prior_samples, aes(x=lambda_b)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab("b") +
  geom_vline(aes(xintercept=ground_truth$lambda_b),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_lambda_b_prior.png")

ggplot(posterior_samples, aes(x=lambda_b)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab("b") +
  geom_vline(aes(xintercept=ground_truth$lambda_b),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_lambda_b_posterior.png")

ggplot(prior_samples, aes(x=exp(lambda_alpha))) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(alpha)) +
  geom_vline(aes(xintercept=exp(ground_truth$lambda_alpha)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_lambda_alpha_prior.png")

ggplot(posterior_samples, aes(x=exp(lambda_alpha))) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(alpha)) +
  geom_vline(aes(xintercept=exp(ground_truth$lambda_alpha)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_lambda_alpha_posterior.png")

#Injection

qplot(
  prior_samples$"Q_gamma",  prior_samples$"Q_b",
  xlab = expression(gamma), ylab = expression(b))
ggsave("scatter_Q_gamma_b_prior.png")

qplot(
  posterior_samples$"Q_gamma", posterior_samples$"Q_b",
  xlab = expression(gamma), ylab = expression(b))
ggsave("scatter_Q_gamma_b_posterior.png")

ggplot(prior_samples[prior_samples$Q_gamma,], aes(x=Q_gamma)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(gamma)) +
  geom_vline(aes(xintercept=ground_truth$Q_gamma),   # Ignore NA values for mean
             color="red", linetype="dashed", size=.75)
ggsave("histogram_Q_gamma_prior.png")

ggplot(posterior_samples, aes(x=Q_gamma)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(gamma)) +
  geom_vline(aes(xintercept=ground_truth$Q_gamma),   # Ignore NA values for mean
             color="red", linetype="dashed", size=.75)
ggsave("histogram_Q_gamma_posterior.png")


ggplot(prior_samples, aes(x=Q_b)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab("b") +
  geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_Q_b_prior.png")


ggplot(posterior_samples, aes(x=Q_b)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab("b") +
  geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_Q_b_posterior.png")

ggplot(prior_samples, aes(x=Q_beta)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(beta)) +
  geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_Q_beta_prior.png")


ggplot(posterior_samples, aes(x=Q_beta)) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(beta)) +
  geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_Q_beta_posterior.png")


ggplot(prior_samples, aes(x=exp(Q_alpha))) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(alpha)) +
  geom_vline(aes(xintercept=exp(ground_truth$Q_alpha)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_Q_alpha_prior.png")


ggplot(posterior_samples[exp(posterior_samples$"Q_alpha") < 50,], aes(x=exp(Q_alpha))) +
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
  theme_gray(base_size = 24) +
  xlab(expression(alpha)) +
  geom_vline(aes(xintercept=exp(ground_truth$Q_alpha)),   # Ignore NA values for mean
             color="red", linetype="dashed", size=0.75)
ggsave("histogram_Q_alpha_posterior.png")

# plot the Kp profile

kp <- read.csv("kp_profile.csv", header = FALSE, col.names = c("time", "Kp"))

ggplot(kp, aes(x=time, y=Kp)) +
  geom_line(size=1.15, linetype=5) +
  theme_gray(base_size = 22)

ggsave("kp_profile.png")

initial <- read.csv("initial_psd.csv", header = FALSE, col.names = c("L", "f"))

ggplot(initial, aes(x=L, y=f)) +
  geom_line(size=1.15, linetype=5) +
  theme_gray(base_size = 22) + ylab(expression(f(0)))

ggsave("initial_psd.png")
