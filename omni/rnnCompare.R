#! /usr/bin/Rscript

library(RSNNS)

args <- commandArgs(trailingOnly = TRUE)
year <- args[1]
setwd("~/Development/PlasmaML/omni/data/")
prefix <- "omni2_"

df <- read.csv(paste(prefix, year, ".dat", sep = ""), 
               header = FALSE, stringsAsFactors = FALSE, 
               colClasses = rep("numeric",55), 
               na.strings = c("99", "999.9", 
                              "9999.", "9.999", "99.99", 
                              "9999", "999999.99", 
                              "99999.99", "9999999."))

trainFeatures <- df[1:7000,c(17, 22, 25)]
trainTargets <- df[1:7000,41]

testFeatures <- df[7001:8760,c(17, 22, 25)]
testTargets <- df[7001:8760,41]
tr <- normalizeData(trainFeatures, type = "0_1")
te <- normalizeData(testFeatures, attr(tr, "normParams"))
trL <- normalizeData(trainTargets, type = "0_1")
teL <- normalizeData(testTargets, attr(trL, "normParams"))

colnames(tr) <- c("Bz", "SigmaBz", "Vsw")
colnames(trL) <- c("Dst")

colnames(te) <- c("Bz", "SigmaBz", "Vsw")
colnames(teL) <- c("Dst")

modelJordan <- jordan(tr, trL, size = c(4),
                      learnFuncParams = c(0.005, 1.75, 0.0005, 4), maxit = 1000,
                      inputsTest = te,
                      targetsTest = teL,
                      linOut = TRUE, learnFunc = "QPTT")

modelEL <- elman(tr, trL, size = c(4),
                 learnFuncParams = c(0.005, 1.75, 0.0005, 4), maxit = 1000,
                 inputsTest = te,
                 targetsTest = teL,
                 linOut = TRUE, learnFunc = "QPTT")


#Now train an MLP based model
modelMLP <- mlp(tr, trL, size = c(5, 3),
                learnFuncParams = c(0.005), maxit = 300,
                inputsTest = te,
                targetsTest = teL,
                linOut = TRUE)

plot(testTargets, type = 'l', main = "Dst prediction", sub = as.character(year), 
     xlab = "Time (Hours)", ylab = "Dst")


lines(denormalizeData(modelJordan$fittedTestValues, attr(trL, "normParams")), col="red")
lines(denormalizeData(modelEL$fittedTestValues, attr(trL, "normParams")), col="green")
lines(denormalizeData(modelMLP$fittedTestValues, attr(trL, "normParams")), col="blue")

hist(modelJordan$fittedTestValues - teL, col="lightblue", breaks=100)
hist(modelMLP$fittedTestValues - teL, col="lightblue", breaks=100)