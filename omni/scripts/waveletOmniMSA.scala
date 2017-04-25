import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.models.neuralnets._

OmniMultiOutputModels.exogenousInputs = List(24, 16)

OmniMultiOutputModels.neuronCounts = List(6, 3)
OmniMultiOutputModels.activations = List(VectorTansig, VectorSigmoid, VectorLinear)

DstMSANNExperiment.learningRate = 0.075
DstMSANNExperiment.momentum = 0.15
DstMSANNExperiment.it = 2000
DstMSANNExperiment.reg = 0.00001


val metrics = DstMSANNExperiment(2, 2, useWavelets = false)

val metricsW = DstMSANNExperiment(2, 2)
