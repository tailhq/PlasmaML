import breeze.linalg.DenseMatrix
import io.github.mandar2812.PlasmaML.omni
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import spire.algebra.Field
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.{RegressionMetrics, BinaryClassificationMetrics}

//First define the experiment parameters
OmniWaveletModels.exogenousInputs = List(24,16,41)
val numVars = OmniWaveletModels.exogenousInputs.length + 1
DstMOGPExperiment.gridSize = 2
DstMOGPExperiment.gridStep = 0.2
OmniWaveletModels.globalOpt = "GS"
DstMOGPExperiment.maxIt = 10

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
val mlpKernel = new MLPKernel(20.0, 5.0)

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

val coRegRBFMatrix = new CoRegRBFKernel(1.7404134335638997)
coRegRBFMatrix.blocked_hyper_parameters = coRegRBFMatrix.hyper_parameters

//Create an adjacency matrix
val adjacencyMatrix = DenseMatrix.tabulate[Double](4,4){(i,j) => coRegCauchyMatrix.evaluate(i,j)}
val graphK = new CoRegGraphKernel(adjacencyMatrix)
graphK.blocked_hyper_parameters =
  graphK.hyper_parameters.filter(h => h.contains("0_0") || h.contains("1_1") || h.contains("2_2") || h.contains("3_3"))

val coRegDiracMatrix = new CoRegDiracKernel

val coRegTMatrix = new CoRegTStudentKernel(1.75)

val kernel: CompositeCovariance[(DenseVector[Double], Int)] =
  (linearK :* mixedEffects) + (tKernel :* coRegRBFMatrix)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix

DstMOGPExperiment.stormAverages = false

DstMOGPExperiment.onsetClassificationScores = true
OmniWaveletModels.threshold = -70.0

val resGPOnset =
  DstMOGPExperiment(2,2,true)(kernel,
    noise).map(
    _.asInstanceOf[BinaryClassificationMetrics].scores_and_labels
  ).map(l =>
    new BinaryClassificationMetrics(l, l.length, true)
  ).toList

resGPOnset.map(_.scores_and_labels).map(sc => sc.map(c => math.pow(c._1 - c._2, 2.0)).sum/sc.length)


val exPred = resGPOnset.last.scores_and_labels

(1 to 100).map(i => {
  val prob = i.toDouble/100.0
  (prob, exPred.map(c => (prob, c._2)).map(c => math.pow(c._1 - c._2, 2.0)).sum/exPred.length)

})


DstMOGPExperiment.onsetClassificationScores = false
val resGP = DstMOGPExperiment(2,2,true)(kernel, noise).map(_.asInstanceOf[RegressionMetrics])

resGP.foreach(_.print)

val resPer = DstPersistenceMOExperiment(2)

//Predictions for an example storm
val (model, scaler) = OmniWaveletModels.train(
  kernel, noise, DstMOGPExperiment.gridSize,
  DstMOGPExperiment.gridStep, useLogSc = false,
  DstMOGPExperiment.maxIt)

OmniWaveletModels.testStart = "2004/11/09/11"
OmniWaveletModels.testEnd = "2004/11/11/09"

val met = OmniWaveletModels.test(model, scaler)
