import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.models.neuralnets._

OmniMultiOutputModels.orderFeat = 3

OmniMultiOutputModels.useWaveletBasis = false
val metrics = OmniMSANN(2e-2, 0.0001, 0.3, 1000, 1.0)

OmniMultiOutputModels.useWaveletBasis = true
val metricsWavelet = OmniMSANN(2e-2, 0.0001, 0.3, 1000, 1.0)


OmniMultiOutputModels.neuronCounts = List(6, 5)
OmniMultiOutputModels.activations = List(VectorTansig, VectorTansig, VectorLinear)

DstMSANNExperiment.learningRate = 0.02
DstMSANNExperiment.momentum = 0.5
DstMSANNExperiment.it = 1000
DstMSANNExperiment.reg = 0.0001

DstMSANNExperiment(4, 3, false)
