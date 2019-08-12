#/usr/bin/Rscript
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)
direc <- args[1]
lossFlag <- args[2]
setwd(direc)

palette2 <- c("firebrick3", "#000000")
palette3 <- c("#CC0C0099", "#5C88DA99")
posterior_samples <- read.csv("posterior_samples.csv", header = TRUE)
prior_samples <- read.csv("prior_samples.csv", header = TRUE)

posterior_samples$sample <- rep("posterior", nrow(posterior_samples))
prior_samples$sample <- rep("prior", nrow(prior_samples))

ground_truth <- read.csv("diffusion_params.csv", header = TRUE, na.strings = c("-Infinity"))

samples <- rbind(prior_samples, posterior_samples)
colnames(samples) <- colnames(posterior_samples)
samples$sample <- as.factor(samples$sample)

if(lossFlag == "loss") {
  qplot(
    prior_samples$"lambda_beta",  prior_samples$"lambda_b",
    xlab = expression(beta), ylab=expression(b))

  ggsave("scatter_beta_b_prior.png")

  qplot(
    posterior_samples$"lambda_beta", posterior_samples$"lambda_b",
    xlab = expression(beta), ylab=expression(b))
  ggsave("scatter_lambda_beta_b_posterior.png")

  qplot(
    prior_samples$"lambda_beta",  prior_samples$"lambda_alpha",
    xlab = expression(beta), ylab=expression(alpha))
  ggsave("scatter_lambda_beta_alpha_prior.png")

  qplot(
    posterior_samples$"lambda_beta", posterior_samples$"lambda_alpha",
    xlab = expression(beta), ylab=expression(alpha))
  ggsave("scatter_lambda_beta_alpha_posterior.png")


  ggplot(samples, aes(x=lambda_beta,y=lambda_b)) +
    geom_point(alpha=0.4, aes(color=sample)) +
    theme_gray(base_size = 24) +
    scale_colour_manual(
      values=palette3, 
      name = "", 
      breaks = levels(samples$sample),
      labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    geom_hline(aes(yintercept=ground_truth$lambda_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    xlab(expression(beta)) + 
    ylab(expression(b)) + 
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("prior_posterior_scatter_lambda_beta_b.png")

  ggplot(samples, aes(x=lambda_alpha,y=lambda_b)) +
    geom_point(alpha=0.5, aes(color=sample)) +
    scale_colour_manual(
      values=palette3, 
      name = "", 
      breaks = levels(samples$sample),
      labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$lambda_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    geom_hline(aes(yintercept=ground_truth$lambda_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) + 
    ylab(expression(b)) + 
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("prior_posterior_scatter_lambda_alpha_b.png")


  ggplot(prior_samples, aes(x=lambda_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(beta)) +
    geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_lambda_beta_prior.png")

  ggplot(posterior_samples, aes(x=lambda_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(beta)) +
    geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_lambda_beta_posterior.png")

  ggplot(samples, aes(x=lambda_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    facet_grid(sample~.) +
    xlab(expression(beta)) +
    geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_lambda_beta.png")

  ggplot(samples, aes(x=lambda_beta)) +
    geom_density(alpha=.2, aes(fill=sample))  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(beta)) +
    scale_fill_manual(
       values= palette3, 
       name = "", 
       breaks = levels(samples$sample),
       labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$lambda_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("density_lambda_beta.png")

  ggplot(prior_samples, aes(x=lambda_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$lambda_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_lambda_b_prior.png")

  ggplot(posterior_samples, aes(x=lambda_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$lambda_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_lambda_b_posterior.png")

  ggplot(samples, aes(x=lambda_b)) +
    geom_density(alpha=.2, aes(fill=sample))  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(b)) +
    scale_fill_manual(
       values= c("#CC0C0099", "#5C88DA99"), 
       name = "", 
       breaks = levels(samples$sample),
       labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$lambda_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("density_lambda_b.png")

  ggplot(samples, aes(x=lambda_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    facet_grid(sample~.) +
    xlab(expression(b)) +
    geom_vline(aes(xintercept=ground_truth$lambda_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_lambda_b.png")

  ggplot(prior_samples, aes(x=lambda_alpha)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) +
    geom_vline(aes(xintercept=ground_truth$lambda_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_lambda_alpha_prior.png")

  ggplot(posterior_samples, aes(x=lambda_alpha)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) +
    geom_vline(aes(xintercept=exp(ground_truth$lambda_alpha)),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_lambda_alpha_posterior.png")

  ggplot(samples, aes(x=lambda_alpha)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    facet_grid(sample~.) +
    xlab(expression(alpha)) +
    geom_vline(aes(xintercept=ground_truth$lambda_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_lambda_alpha.png")

  ggplot(samples, aes(x=lambda_alpha)) +
    geom_density(alpha=.2, aes(fill=sample))  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) +
    scale_fill_manual(
       values= c("#CC0C0099", "#5C88DA99"), 
       name = "", 
       breaks = levels(samples$sample),
       labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$lambda_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("density_lambda_alpha.png")


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

} else {

  qplot(
    prior_samples$"Q_beta",  prior_samples$"Q_b",
    xlab = expression(beta), ylab=expression(b))

  ggsave("scatter_Q_beta_b_prior.png")

  qplot(
    posterior_samples$"Q_beta", posterior_samples$"Q_b",
    xlab = expression(beta), ylab=expression(b))
  ggsave("scatter_Q_beta_b_posterior.png")

  qplot(
    prior_samples$"Q_beta",  prior_samples$"Q_alpha",
    xlab = expression(beta), ylab=expression(alpha))
  ggsave("scatter_Q_beta_alpha_prior.png")

  qplot(
    posterior_samples$"Q_beta", posterior_samples$"Q_alpha",
    xlab = expression(beta), ylab=expression(alpha))
  ggsave("scatter_Q_beta_alpha_posterior.png")


  ggplot(samples, aes(x=Q_beta,y=Q_b)) +
    geom_point(alpha=0.4, aes(color=sample)) +
    theme_gray(base_size = 24) +
    scale_colour_manual(
      values=palette3, 
      name = "", 
      breaks = levels(samples$sample),
      labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    geom_hline(aes(yintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    xlab(expression(beta)) + 
    ylab(expression(b)) + 
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("prior_posterior_scatter_Q_beta_b.png")

  ggplot(samples, aes(x=Q_alpha,y=Q_b)) +
    geom_point(alpha=0.5, aes(color=sample)) +
    scale_colour_manual(
      values=palette3, 
      name = "", 
      breaks = levels(samples$sample),
      labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$Q_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    geom_hline(aes(yintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) + 
    ylab(expression(b)) + 
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("prior_posterior_scatter_Q_alpha_b.png")


  ggplot(prior_samples, aes(x=Q_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(beta)) +
    geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_Q_beta_prior.png")

  ggplot(posterior_samples, aes(x=Q_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(beta)) +
    geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_Q_beta_posterior.png")

  ggplot(samples, aes(x=Q_beta)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    facet_grid(sample~.) +
    xlab(expression(beta)) +
    geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_Q_beta.png")

  ggplot(samples, aes(x=Q_beta)) +
    geom_density(alpha=.2, aes(fill=sample))  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(beta)) +
    scale_fill_manual(
       values= palette3, 
       name = "", 
       breaks = levels(samples$sample),
       labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$Q_beta),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("density_Q_beta.png")

  ggplot(prior_samples, aes(x=Q_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_Q_b_prior.png")

  ggplot(posterior_samples, aes(x=Q_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab("b") +
    geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_Q_b_posterior.png")

  ggplot(samples, aes(x=Q_b)) +
    geom_density(alpha=.2, aes(fill=sample))  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(b)) +
    scale_fill_manual(
       values= c("#CC0C0099", "#5C88DA99"), 
       name = "", 
       breaks = levels(samples$sample),
       labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("density_Q_b.png")

  ggplot(samples, aes(x=Q_b)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    facet_grid(sample~.) +
    xlab(expression(b)) +
    geom_vline(aes(xintercept=ground_truth$Q_b),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_Q_b.png")

  ggplot(prior_samples, aes(x=Q_alpha)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) +
    geom_vline(aes(xintercept=ground_truth$Q_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_Q_alpha_prior.png")

  ggplot(posterior_samples, aes(x=Q_alpha)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) +
    geom_vline(aes(xintercept=exp(ground_truth$Q_alpha)),   # Ignore NA values for mean
               color="black", linetype="dashed", size=0.75)
  ggsave("histogram_Q_alpha_posterior.png")

  ggplot(samples, aes(x=Q_alpha)) +
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   binwidth=.5,
                   colour="black", fill="white") +
    geom_density(alpha=.2, fill="#FF6666")  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    facet_grid(sample~.) +
    xlab(expression(alpha)) +
    geom_vline(aes(xintercept=ground_truth$Q_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75)
  ggsave("histogram_Q_alpha.png")

  ggplot(samples, aes(x=Q_alpha)) +
    geom_density(alpha=.2, aes(fill=sample))  +# Overlay with transparent density plot
    theme_gray(base_size = 24) +
    xlab(expression(alpha)) +
    scale_fill_manual(
       values= c("#CC0C0099", "#5C88DA99"), 
       name = "", 
       breaks = levels(samples$sample),
       labels=c("posterior", "prior")) +
    geom_vline(aes(xintercept=ground_truth$Q_alpha),   # Ignore NA values for mean
               color="black", linetype="dashed", size=.75) +
    theme(legend.position="top", legend.direction = "horizontal")

  ggsave("density_Q_alpha.png")
  
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
  

}

