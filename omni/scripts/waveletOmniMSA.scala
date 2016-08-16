import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels._

OmniWaveletModels.useWaveletBasis = false
OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0)

OmniWaveletModels.useWaveletBasis = true
OmniWaveletModels(2e-1, 0.0, 0.0, 30, 1.0)


// Increase the number of features to boost 6 hr predictions
OmniWaveletModels.orderFeat = 4

OmniWaveletModels(4e-1, 0.01, 0.4, 20, 1.0)

OmniWaveletModels.useWaveletBasis = false
OmniWaveletModels(4e-2, 0.0, 0.2, 20, 1.0)


val linearK = new PolynomialKernel(1, 0.0)
val rbfK = new RationalQuadraticKernel(1.5, 1.5)
//rbfK.blocked_hyper_parameters = List("mu")
linearK.blocked_hyper_parameters = List("degree", "offset")

val d = new DiracKernel(0.3)
d.blocked_hyper_parameters = List("noiseLevel")

val n = new CoRegRBFKernel(0.5)
n.blocked_hyper_parameters = n.hyper_parameters

val k = new CoRegRBFKernel(1.5)
val k1 = new CoRegDiracKernel

val kernel = (linearK :* k) + (rbfK :* k)
val noise = d :* n

OmniWaveletModels.orderFeat = 4
OmniWaveletModels.orderTarget = 2

val (model, sc) = OmniWaveletModels.train(kernel, noise, 3, 0.2, false)

OmniWaveletModels.test(model, sc)
