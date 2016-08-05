import io.github.mandar2812.PlasmaML.omni.{CoRegKernel, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels.{DiracKernel, RBFKernel, RationalQuadraticKernel}

OmniWaveletModels.useWaveletBasis = false
OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0)

OmniWaveletModels.useWaveletBasis = true
OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0)


// Increase the number of features to boost 6 hr predictions
OmniWaveletModels.orderFeat = 4

OmniWaveletModels(4e-1, 0.01, 0.4, 20, 1.0)

OmniWaveletModels.useWaveletBasis = false
OmniWaveletModels(4e-2, 0.0, 0.2, 20, 1.0)


val rbf = new PolynomialKernel(1, 1.0)
val rat = new RationalQuadraticKernel(1.2, 1.0)
rat.blocked_hyper_parameters = List("mu")
rbf.blocked_hyper_parameters = List("degree")

val d = new DiracKernel(1.0)

val n = new CoRegKernel
val k = new CoRegKernel

val kernel = (k :*: rbf) + (k :*: rat)
val noise = n :*: d

OmniWaveletModels.orderFeat = 4
OmniWaveletModels.orderTarget = 2

val (model, sc) = OmniWaveletModels.train(kernel, noise, 3, 0.2, false)

OmniWaveletModels.test(model, sc)
