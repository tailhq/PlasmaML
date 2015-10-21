#! /usr/bin/Rscript

library(RSNNS)
library(caret)

trainFeatures <- df[1:5000,c(10, 17, 25)]
trainTargets <- df[1:5000,41]

testFeatures <- df[5001:8760,c(10, 17, 25)]
testTargets <- df[5001:8760,41]

modelEL <- elman(trainFeatures, trainTargets, size = c(8), 
                 learnFuncParams = c(0.1), maxit = 500, 
                 inputsTest = testFeatures, 
                 targetsTest = testTargets, 
                 linOut = FALSE)

modelMLP <- mlp(x = trainFeatures, y = trainTargets, 
                inputsTest = testFeatures, linOut = TRUE, maxit = 500, 
                targetsTest = testTargets, size = 4, 
                hiddenActFunc = "Act_Logistic")



plot(testTargets, type = 'l')

lines(modelEL$fittedTestValues,col="blue")

hist(modelEL$fittedTestValues - testTargets, col="lightblue", breaks=100)


lines(modelMLP$fittedTestValues,col="red")