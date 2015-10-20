#! /usr/bin/Rscript

library(RSNNS)
library(caret)

trainFeatures <- preProcess(as.matrix(df[1:5000,c(17,22,25,26,27)]))
trainTargets <- preProcess(as.matrix(df[1:5000,41]))

testFeatures <- preProcess(as.matrix(df[5001:8760,c(17,22,25,26,27)]))
testTargets <- preProcess(as.matrix(df[5001:8760,41]))

modelEL <- jordan(x = trainFeatures$data, y = trainTargets,
                 inputsTest = testFeatures, learnFunc = "JE_BP_Momentum",
                 targetsTest = testTargets, size = 10, maxit = 500)


plot(x = as.double(1:length(testTargets)), y = testTargets, type = 'l')

lines(as.double(1:length(testTargets)),modelEL$fittedTestValues,col="blue")


plotRegressionError(testTargets, modelEL$fittedTestValues)

hist(modelEL$fittedTestValues - testTargets, col="lightblue", breaks=100)
