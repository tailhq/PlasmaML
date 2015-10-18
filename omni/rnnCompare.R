#! /usr/bin/Rscript

library(RSNNS)

trainFeatures <- df[1:5000,c(17,22,25,26,27)]
trainTargets <- df[1:5000,41]

testFeatures <- df[5001:8760,c(17,22,25,26,27)]
testTargets <- df[5001:8760,41]

modelEL <- jordan(x = trainFeatures, y = trainTargets,
                 inputsTest = testFeatures, learnFunc = "JE_BP_Momentum",
                 targetsTest = testTargets, size = 4)

plotRegressionError(testTargets, modelEL$fittedTestValues)

hist(modelEL$fittedTestValues - testTargets, col="lightblue")