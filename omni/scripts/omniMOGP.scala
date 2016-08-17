import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels._

val linearK = new PolynomialKernel(1, 0.0)
val cauKernel = new CauchyKernel(1.5)

linearK.blocked_hyper_parameters = linearK.hyper_parameters

val d = new DiracKernel(0.32)
d.blocked_hyper_parameters = List("noiseLevel")

val coRegCauchyMatrix = new CoRegCauchyKernel(3.5)
coRegCauchyMatrix.blocked_hyper_parameters = coRegCauchyMatrix.hyper_parameters

val coRegLaplaceMatrix = new CoRegLaplaceKernel(5.0)
coRegLaplaceMatrix.blocked_hyper_parameters = coRegLaplaceMatrix.hyper_parameters
val k1 = new CoRegDiracKernel

val waveletF = (x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0)

val waveletKernel = new WaveletKernel(waveletF)(3.5)


val kernel: CompositeCovariance[(DenseVector[Double], Int)] = (linearK :* coRegLaplaceMatrix) + (cauKernel :* coRegCauchyMatrix) + (waveletKernel :* coRegCauchyMatrix)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* k1

OmniWaveletModels.exogenousInputs = List(16,24)

DstMOGPExperiment.gridSize = 4

DstMOGPExperiment(3,2,true)(kernel, noise)
