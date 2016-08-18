import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels._

val linearK = new PolynomialKernel(1, 0.0)
val cauKernel = new CauchyKernel(6.2)

linearK.blocked_hyper_parameters = linearK.hyper_parameters

val d = new DiracKernel(0.32)
d.blocked_hyper_parameters = List("noiseLevel")

val coRegCauchyMatrix = new CoRegCauchyKernel(5.5)
coRegCauchyMatrix.blocked_hyper_parameters = coRegCauchyMatrix.hyper_parameters

val coRegLaplaceMatrix = new CoRegLaplaceKernel(5.0)
coRegLaplaceMatrix.blocked_hyper_parameters = coRegLaplaceMatrix.hyper_parameters

val coRegRBFMatrix = new CoRegRBFKernel(1.0)
coRegRBFMatrix.blocked_hyper_parameters = coRegRBFMatrix.hyper_parameters

val coRegDiracMatrix = new CoRegDiracKernel

val waveletF = (x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0)

val waveletKernel = new WaveletKernel(waveletF)(6.2)


val kernel: CompositeCovariance[(DenseVector[Double], Int)] = (linearK :* coRegLaplaceMatrix) + (d:*coRegCauchyMatrix) /*+ (cauKernel :* coRegCauchyMatrix) + (waveletKernel :* coRegCauchyMatrix)*/

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix

OmniWaveletModels.exogenousInputs = List(24,16,41)

DstMOGPExperiment.gridSize = 4

DstMOGPExperiment(3,2,false)(kernel, noise)
