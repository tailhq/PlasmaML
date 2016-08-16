import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels._

val linearK = new PolynomialKernel(1, 0.0)
val fbmK = new DiracKernel(0.1)
val rbfKernel = new CauchyKernel(2.5)

fbmK.blocked_hyper_parameters = fbmK.hyper_parameters
linearK.blocked_hyper_parameters = List("degree", "offset")

val d = new DiracKernel(0.92)
//d.blocked_hyper_parameters = List("noiseLevel")

val n = new CoRegRBFKernel(50)
n.blocked_hyper_parameters = n.hyper_parameters

val k = new CoRegLaplaceKernel(5.2)
val k1 = new CoRegDiracKernel

val kernel: CompositeCovariance[(DenseVector[Double], Int)] = (linearK :* n) //+ (rbfKernel :* n)
val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* k1

OmniWaveletModels.exogenousInputs = List(16,24)

DstMOGPExperiment(6,2,true)(kernel, noise)
