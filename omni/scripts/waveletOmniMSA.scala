import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.models.neuralnets._

OmniMultiOutputModels.exogenousInputs = List(24, 16)

OmniMultiOutputModels.neuronCounts = List(6, 5)
OmniMultiOutputModels.activations = List(VectorTansig, VectorTansig, VectorLinear)

DstMSANNExperiment.learningRate = 0.02
DstMSANNExperiment.momentum = 0.5
DstMSANNExperiment.it = 2000
DstMSANNExperiment.reg = 0.0001

DstMSANNExperiment(2, 2, true)
