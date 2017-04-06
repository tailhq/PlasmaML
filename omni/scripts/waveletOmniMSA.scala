import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.models.neuralnets._

OmniMultiOutputModels.exogenousInputs = List(24, 16)

OmniMultiOutputModels.neuronCounts = List(18, 12, 6)
OmniMultiOutputModels.activations = List(VectorTansig, VectorSigmoid, VectorTansig, VectorLinear)

DstMSANNExperiment.learningRate = 0.02
DstMSANNExperiment.momentum = 0.7
DstMSANNExperiment.it = 2000
DstMSANNExperiment.reg = 0.0001

val metricsW = DstMSANNExperiment(3, 2, true)

val metrics = DstMSANNExperiment(3, 2, false)
