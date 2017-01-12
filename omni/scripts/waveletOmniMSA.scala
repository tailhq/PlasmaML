import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniMultiOutputModels}
import io.github.mandar2812.dynaml.kernels._

OmniMultiOutputModels.useWaveletBasis = false
OmniMultiOutputModels(2e-1, 0.0, 0.0, 30, 1.0)

OmniMultiOutputModels.useWaveletBasis = true
OmniMultiOutputModels(2e-1, 0.0, 0.0, 30, 1.0)


// Increase the number of features to boost 6 hr predictions
OmniMultiOutputModels.orderFeat = 3

OmniMultiOutputModels(4e-2, 0.01, 0.4, 20, 1.0)

OmniMultiOutputModels.useWaveletBasis = false
OmniMultiOutputModels(4e-2, 0.0, 0.2, 20, 1.0)


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

OmniMultiOutputModels.orderFeat = 4
OmniMultiOutputModels.orderTarget = 2

val (model, sc) = OmniMultiOutputModels.train(kernel, noise, 3, 0.2, false)

OmniMultiOutputModels.test(model, sc)
