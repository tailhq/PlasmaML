#! /usr/bin/Rscript
args <- commandArgs(trailingOnly = TRUE)
year <- args[1]
setwd("~/Development/PlasmaML/omni/data/")
prefix <- "omni2_"
system(paste("wget ftp://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/",
             prefix, year, ".dat", sep = ""))
system(paste("sed -i 's/ \\{1,\\}/,/g' ", prefix, year, ".dat", sep = ""))
