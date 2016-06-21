import io.github.mandar2812.PlasmaML.omni.OmniWaveletModels

OmniWaveletModels.useWaveletBasis = false
OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0)

OmniWaveletModels.useWaveletBasis = true
OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0)


// Increase the number of features to boost 6 hr predictions
OmniWaveletModels.orderFeat = 4

OmniWaveletModels(4e-1, 0.01, 0.4, 20, 1.0)

OmniWaveletModels.useWaveletBasis = false
OmniWaveletModels(4e-2, 0.0, 0.2, 20, 1.0)