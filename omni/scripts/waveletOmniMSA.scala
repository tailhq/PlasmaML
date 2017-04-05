import io.github.mandar2812.PlasmaML.omni._

OmniMultiOutputModels.orderFeat = 3

OmniMultiOutputModels.useWaveletBasis = false
val metrics = OmniMSANN(2e-2, 0.0001, 0.3, 1000, 1.0)

OmniMultiOutputModels.useWaveletBasis = true
val metricsWavelet = OmniMSANN(2e-2, 0.0001, 0.3, 1000, 1.0)
