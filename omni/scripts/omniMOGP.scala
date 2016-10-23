import breeze.linalg.DenseMatrix
import io.github.mandar2812.PlasmaML.omni
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniWaveletModels}
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import spire.algebra.Field
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.{BinaryClassificationMetrics, RegressionMetrics}

//First define the experiment parameters
OmniWaveletModels.exogenousInputs = List(24,16)
val numVars = OmniWaveletModels.exogenousInputs.length + 1
DstMOGPExperiment.gridSize = 2
DstMOGPExperiment.gridStep = 0.2
OmniWaveletModels.globalOpt = "CSA"
DstMOGPExperiment.maxIt = 10


OmniWaveletModels.orderFeat = 3
OmniWaveletModels.orderTarget = 2

val num_features = if(OmniWaveletModels.deltaT.isEmpty) {
  (1 to numVars).map(_ => math.pow(2.0, OmniWaveletModels.orderFeat)).sum.toInt
} else {
  OmniWaveletModels.deltaT.sum
}

val linearK = new PolynomialKernel(1, 0.0)
linearK.blocked_hyper_parameters = linearK.hyper_parameters

//Create a Vector Field of the appropriate dimension so that
//we can create stationary kernels
implicit val ev = VectorField(num_features)

val tKernel = new TStudentKernel(0.5+1.0/num_features)
//tKernel.blocked_hyper_parameters = tKernel.hyper_parameters
val mlpKernel = new MLPKernel(20.0/num_features.toDouble, 1.382083995440671)


val d = new DiracKernel(0.037)
d.blocked_hyper_parameters = d.hyper_parameters

val mixedEffects = new MixedEffectRegularizer(0.0)
mixedEffects.blocked_hyper_parameters = mixedEffects.hyper_parameters

val mixedEffects2 = new MixedEffectRegularizer(0.5)
mixedEffects2.blocked_hyper_parameters = mixedEffects2.hyper_parameters

val coRegCauchyMatrix = new CoRegCauchyKernel(10.0)
coRegCauchyMatrix.blocked_hyper_parameters = coRegCauchyMatrix.hyper_parameters

val coRegLaplaceMatrix = new CoRegLaplaceKernel(10.0)

val coRegRBFMatrix = new CoRegRBFKernel(30.33509292873575)
coRegRBFMatrix.blocked_hyper_parameters = coRegRBFMatrix.hyper_parameters

//Create an adjacency matrix
//val adjacencyMatrix = DenseMatrix.tabulate[Double](4,4){(i,j) => coRegCauchyMatrix.evaluate(i,j)}
//val graphK = new CoRegGraphKernel(adjacencyMatrix)
//graphK.blocked_hyper_parameters =
//  graphK.hyper_parameters.filter(h => h.contains("0_0") || h.contains("1_1") || h.contains("2_2") || h.contains("3_3"))

val coRegDiracMatrix = new CoRegDiracKernel

val coRegTMatrix = new CoRegTStudentKernel(1.75)

val kernel: CompositeCovariance[(DenseVector[Double], Int)] =
  (linearK :* mixedEffects) + (tKernel :* coRegRBFMatrix) + (mlpKernel :* mixedEffects2)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix

DstMOGPExperiment.stormAverages = false

DstMOGPExperiment.onsetClassificationScores = true
OmniWaveletModels.threshold = -70.0


//Calculate brier scores on the 63 storms set.
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

//Calculate Regression scores on the 63 storms data set
DstMOGPExperiment.onsetClassificationScores = false
val resGP = DstMOGPExperiment(4,2,true)(kernel, noise).map(_.asInstanceOf[RegressionMetrics])

resGP.foreach(_.print)

//Calculate regression scores of the persistence model
val resPer = DstPersistenceMOExperiment(2)


//Generating sample predictions and error bars for the halloween storm
val kernel: CompositeCovariance[(DenseVector[Double], Int)] =
  (linearK :* mixedEffects) + (tKernel :* coRegRBFMatrix) + (mlpKernel :* coRegCauchyMatrix)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix


OmniWaveletModels.globalOpt = "GS"
DstMOGPExperiment.gridSize = 2
DstMOGPExperiment.gridStep = 0.2
OmniWaveletModels.orderFeat = 2
OmniWaveletModels.orderTarget = 2
//Predictions for an example storm
OmniWaveletModels.numStorms = 5
val (model, scaler) = OmniWaveletModels.trainStorms(
  kernel, noise, DstMOGPExperiment.gridSize,
  DstMOGPExperiment.gridStep, useLogSc = true,
  DstMOGPExperiment.maxIt)

model.persist()

OmniWaveletModels.testStart = "2003/11/20/00"
OmniWaveletModels.testEnd = "2003/11/22/00"

(0 to 3).foreach(i => {
  val met = OmniWaveletModels.generatePredictions(model, scaler, i).map(c => c._1.toString+","+c._2.toString+","+c._3.toString+","+c._4.toString)

  val onsetScores = OmniWaveletModels.generateOnsetPredictions(model, scaler, i).map(c => c._1.toString+","+c._2.toString)

  DynaMLPipe.streamToFile("data/mogp_preds_errorbars"+i+".csv")(met)

  DynaMLPipe.streamToFile("data/mogp_onset_predictions"+i+".csv")(onsetScores.toStream)

})
