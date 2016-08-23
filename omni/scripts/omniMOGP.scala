import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.kernels._

val linearK = new PolynomialKernel(1, 0.0)
val expKernel = new ExponentialKernel(1.5)
val cauKernel = new CauchyKernel(6.2)
val rBFKernel = new SEKernel(30.0, 2.5)
val tKernel = new TStudentKernel(1.5)

linearK.blocked_hyper_parameters = linearK.hyper_parameters

val d = new DiracKernel(0.72)
//d.blocked_hyper_parameters = List("noiseLevel")

val mixedEffects = new MixedEffectRegularizer(1.0)
mixedEffects.blocked_hyper_parameters = mixedEffects.hyper_parameters

val coRegCauchyMatrix = new CoRegCauchyKernel(5.5)
//coRegCauchyMatrix.blocked_hyper_parameters = coRegCauchyMatrix.hyper_parameters

val coRegLaplaceMatrix = new CoRegLaplaceKernel(5.0)
coRegLaplaceMatrix.blocked_hyper_parameters = coRegLaplaceMatrix.hyper_parameters

val coRegRBFMatrix = new CoRegRBFKernel(1.0)
coRegRBFMatrix.blocked_hyper_parameters = coRegRBFMatrix.hyper_parameters

val coRegDiracMatrix = new CoRegDiracKernel

val waveletF = (x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0)

val waveletKernel = new WaveletKernel(waveletF)(6.2)


val kernel: CompositeCovariance[(DenseVector[Double], Int)] = (linearK :* mixedEffects) + (tKernel :* coRegCauchyMatrix)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix

OmniWaveletModels.exogenousInputs = List(24,16,41)

DstMOGPExperiment.gridSize = 2
OmniWaveletModels.globalOpt = "CSA"
DstMOGPExperiment.maxIt = 25

DstMOGPExperiment(3,2,true)(kernel, noise)
