import spire.algebra.Field

import scalaxy.streams.optimize
import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.PlasmaML.omni.DstPersistenceMOExperiment
//DynaML Imports: Data pipes and Encoders
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.pipes.{Encoder, Reducer}
import io.github.mandar2812.dynaml.pipes.Reducer._
//Import required DynaML components like kernels, evaluation metrics.
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.DynaMLPipe._
//Import the OMNI experiments
import io.github.mandar2812.PlasmaML.omni
import io.github.mandar2812.PlasmaML.omni.{DstMOGPExperiment, OmniMultiOutputModels}


//First define the experiment parameters
OmniMultiOutputModels.exogenousInputs = List(24,16)
val numVars = OmniMultiOutputModels.exogenousInputs.length + 1
OmniMultiOutputModels.globalOpt = "GS"
DstMOGPExperiment.gridSize = 3
DstMOGPExperiment.gridStep = 0.22
OmniMultiOutputModels.orderFeat = 2
OmniMultiOutputModels.orderTarget = 2
//Predictions for an example storm
OmniMultiOutputModels.numStorms = 12
//Calculate the number of features
//from the provided lengths of each
//input like Dst, V, Bz, ... etc
val num_features = if(OmniMultiOutputModels.deltaT.isEmpty) {
  (1 to numVars).map(_ => math.pow(2.0, OmniMultiOutputModels.orderFeat)).sum.toInt
} else {
  OmniMultiOutputModels.deltaT.sum
}
//Create a Vector Field of the appropriate dimension so that
//we can create stationary kernels
implicit val ev = VectorField(num_features)

//Create some feature split schemes
//so we can create a number of decomposable
//covariances
val size_split = math.pow(2.0, OmniMultiOutputModels.orderFeat).toInt
val sp = breezeDVSplitEncoder(size_split)

implicit val sp2 = Encoder(
  (x: (DenseVector[Double], Int)) => {
    optimize {
      val vecs = Array.tabulate(x._1.length/size_split)(i => x._1(i*size_split until math.min((i+1)*size_split, x._1.length)))
      vecs.map(xs => (xs, x._2))
    }
  },
  (xs: Array[(DenseVector[Double], Int)]) => optimize{(DenseVector(xs.map(_._1.toArray).reduceLeft(_++_)), xs.head._2)}
)

DstMOGPExperiment.stormAverages = false
val linearK = new PolynomialKernel(1, 0.0)
linearK.block_all_hyper_parameters

val quadKernel = new FBMKernel(0.8)//+DstMOGPExperiment.gridStep)
quadKernel.block_all_hyper_parameters

val d = new DiracKernel(0.037)
d.blocked_hyper_parameters = d.hyper_parameters

val tKernel = new TStudentKernel(1.0/*0.5+1.0/num_features*/)
//tKernel.block_all_hyper_parameters

val mlpKernel = new MLPKernel(0.909, 0.909)
mlpKernel.block_all_hyper_parameters

val coRegCauchyMatrix = new CoRegCauchyKernel(10.0)
coRegCauchyMatrix.block_all_hyper_parameters

val coRegLaplaceMatrix = new CoRegLaplaceKernel(9.8)
coRegLaplaceMatrix.block_all_hyper_parameters

val coRegDiracMatrix = new CoRegDiracKernel

val coRegTMatrix = new CoRegTStudentKernel(10.2)
coRegTMatrix.block_all_hyper_parameters

val dstKernel = new DecomposableCovariance(
  linearK + tKernel,
  linearK + mlpKernel,
  linearK + quadKernel
)(sp)

//val kernel = dstKernel :* (coRegLaplaceMatrix)

val kernel = new DecomposableCovariance(
  linearK :* coRegLaplaceMatrix,
  (quadKernel :* coRegTMatrix) + (tKernel :* coRegTMatrix),
  quadKernel :* coRegTMatrix,
  mlpKernel :* coRegCauchyMatrix)

//val kernel = (linearK :* coRegLaplaceMatrix) + (tKernel :* coRegTMatrix) + (mlpKernel :* coRegLaplaceMatrix)

val noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegCauchyMatrix

OmniMultiOutputModels.useWaveletBasis = true

val (stp_model, stp_scaler) = OmniMultiOutputModels.trainSTPStorms(
  linearK :* coRegLaplaceMatrix, noise, 5.5, DstMOGPExperiment.gridSize,
  DstMOGPExperiment.gridStep, useLogSc = true,
  DstMOGPExperiment.maxIt)


/*val storm0 = ("2000/04/06/08", "2000/04/08/09")
val storm1 = ("2001/03/30/04", "2001/04/01/21")
val storm2 = ("2001/04/11/08", "2001/04/13/07")
val halloween2003Storm = ("2003/11/20/00", "2003/11/22/00")
val storm2004 = ("2004/11/07/10", "2004/11/08/21")

val storms = Seq(storm0, storm1, storm2, halloween2003Storm, storm2004)

storms.foreach(s => {
  println("\nGenerating plots for storm: "+s._1+" to "+s._2)
  OmniWaveletModels.testStart = s._1
  OmniWaveletModels.testEnd = s._2

  (0 to 3).foreach(i => {
    val met = OmniWaveletModels.generatePredictions(model, scaler, i).map(c => c._1.toString+","+c._2.toString+","+c._3.toString+","+c._4.toString)

    val onsetScores = OmniWaveletModels.generateOnsetPredictions(model, scaler, i).map(c => c._1.toString+","+c._2.toString)

    DynaMLPipe.streamToFile("data/mogp_preds_"+OmniWaveletModels.testStart.split("/").take(3).mkString("_")+"-"+i+".csv")(met)

    DynaMLPipe.streamToFile("data/mogp_onset_"+OmniWaveletModels.testStart.split("/").take(3).mkString("_")+"-"+i+".csv")(onsetScores.toStream)

  })
})*/

//Calculate Regression scores on the 63 storms data set
DstMOGPExperiment.onsetClassificationScores = false

val resSTP = DstMOGPExperiment.testRegression(stp_model, stp_scaler).map(_.asInstanceOf[RegressionMetrics])
//Print the Regression evaluation results to the console
resSTP.foreach(_.print)


//Now calculate brier scores on the 63 storms data
/*DstMOGPExperiment.onsetClassificationScores = true
OmniWaveletModels.threshold = -70.0
val resGPOnset =
  DstMOGPExperiment.test(model, scaler).map(
    _.asInstanceOf[BinaryClassificationMetrics].scores_and_labels
  ).map(l =>
    new BinaryClassificationMetrics(l, l.length, true)
  ).toList

resGPOnset.map(_.scores_and_labels).map(sc => sc.map(c => math.pow(c._1 - c._2, 2.0)).sum/sc.length)

val exPred = resGPOnset.last.scores_and_labels

val brier = (1 to 100).map(i => {
  val prob = i.toDouble/100.0
  (prob, exPred.map(c => (prob, c._2)).map(c => math.pow(c._1 - c._2, 2.0)).sum/exPred.length)

})
//Write the brier scores to a csv file
DynaMLPipe.streamToFile("data/brier_scores.csv")(brier.map(c => c._1.toString+","+c._2.toString).toStream)

//Calculate regression scores of the persistence model
val resPer = DstPersistenceMOExperiment(2)
*/
