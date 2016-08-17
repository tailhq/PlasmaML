import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels._

val linearK = new PolynomialKernel(1, 0.0)
val fbmK = new DiracKernel(0.1)
val cauKernel = new CauchyKernel(1.5)

fbmK.blocked_hyper_parameters = fbmK.hyper_parameters
linearK.blocked_hyper_parameters = linearK.hyper_parameters

val d = new DiracKernel(0.32)
d.blocked_hyper_parameters = List("noiseLevel")

val n = new CoRegCauchyKernel(3.5)
n.blocked_hyper_parameters = n.hyper_parameters

val k = new CoRegLaplaceKernel(5.0)
k.blocked_hyper_parameters = k.hyper_parameters
val k1 = new CoRegDiracKernel

val kernel: CompositeCovariance[(DenseVector[Double], Int)] = (linearK :* k) + (cauKernel :* n)
val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* k1

OmniWaveletModels.exogenousInputs = List(16,24)

DstMOGPExperiment.gridSize = 4

DstMOGPExperiment(3,2,true)(kernel, noise)
