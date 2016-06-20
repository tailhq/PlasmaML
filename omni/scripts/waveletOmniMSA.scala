import io.github.mandar2812.PlasmaML.omni.OmniWaveletModels

OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0, useWaveletBasis = false)

OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0, useWaveletBasis = true)


// Increase the number of features to boost 6 hr predictions
OmniWaveletModels.orderFeat = 4

OmniWaveletModels(4e-1, 0.01, 0.4, 20, 1.0, useWaveletBasis = true)
