import io.github.mandar2812.PlasmaML.omni.OmniWaveletModels

OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0, useWaveletBasis = false)

OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0, useWaveletBasis = true)


// Increase the number of features to boost 6 hr predictions
OmniWaveletModels.orderFeat = 4

OmniWaveletModels(2e-1, 0.0, 0.2, 40, 1.0, useWaveletBasis = true)
