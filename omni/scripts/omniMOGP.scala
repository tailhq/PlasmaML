import breeze.linalg.DenseMatrix
import io.github.mandar2812.PlasmaML.omni
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import spire.algebra.Field
import io.github.mandar2812.dynaml.analysis.VectorField

//First define the experiment parameters
OmniWaveletModels.exogenousInputs = List(24,16,41)
val numVars = OmniWaveletModels.exogenousInputs.length + 1
DstMOGPExperiment.gridSize = 2
DstMOGPExperiment.gridStep = 0.2
OmniWaveletModels.globalOpt = "CSA"
DstMOGPExperiment.maxIt = 5

val num_features = if(OmniWaveletModels.deltaT.isEmpty) {
  (1 to numVars).map(_ => math.pow(2.0, OmniWaveletModels.orderFeat)).sum.toInt
} else {
  OmniWaveletModels.deltaT.sum
}

//Create a Vector Field of the appropriate dimension so that
//we can create stationary kernels
implicit val ev = VectorField(num_features)

val linearK = new PolynomialKernel(1, 0.0)
val cauKernel = new CauchyKernel(6.2)
val rBFKernel = new SEKernel(1.5, 1.5)
val tKernel = new TStudentKernel(0.1341331567117945)
tKernel.blocked_hyper_parameters = tKernel.hyper_parameters
val fbmK = new FBMKernel(1.0)

fbmK.blocked_hyper_parameters = fbmK.hyper_parameters
linearK.blocked_hyper_parameters = linearK.hyper_parameters

val d = new DiracKernel(0.037)
d.blocked_hyper_parameters = d.hyper_parameters

val mixedEffects = new MixedEffectRegularizer(0.0)
mixedEffects.blocked_hyper_parameters = mixedEffects.hyper_parameters

val mixedEffects2 = new MixedEffectRegularizer(0.5)
mixedEffects2.blocked_hyper_parameters = mixedEffects2.hyper_parameters

val coRegCauchyMatrix = new CoRegCauchyKernel(10.0)
coRegCauchyMatrix.blocked_hyper_parameters = coRegCauchyMatrix.hyper_parameters

val coRegLaplaceMatrix = new CoRegLaplaceKernel(10.0)

val coRegRBFMatrix = new CoRegRBFKernel(8.0)
coRegRBFMatrix.blocked_hyper_parameters = coRegRBFMatrix.hyper_parameters

//Create an adjacency matrix
val adjacencyMatrix = DenseMatrix.tabulate[Double](4,4){(i,j) => coRegCauchyMatrix.evaluate(i,j)}
val graphK = new CoRegGraphKernel(adjacencyMatrix)
graphK.blocked_hyper_parameters =
  graphK.hyper_parameters.filter(h => h.contains("0_0") || h.contains("1_1") || h.contains("2_2") || h.contains("3_3"))

val coRegDiracMatrix = new CoRegDiracKernel

val coRegTMatrix = new CoRegTStudentKernel(1.75)

val kernel: CompositeCovariance[(DenseVector[Double], Int)] =
  (linearK :* mixedEffects) + (tKernel :* graphK)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix

DstMOGPExperiment.stormAverages = false
val resGP = DstMOGPExperiment(2,2,true)(kernel, noise)
val resPer = DstPersistenceMOExperiment(2)
