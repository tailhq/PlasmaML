import breeze.linalg._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
//DynaML imports
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.pipes._
//Import Omni programs
import io.github.mandar2812.PlasmaML.omni._

val polynomialKernel = new PolynomialKernel(1, 0.5)
polynomialKernel.block("degree")

implicit val ev = VectorField(OmniOSA.input_dimensions)

val tKernel = new TStudentKernel(0.01)
tKernel.block_all_hyper_parameters

val rbfKernel = new RBFKernel(1.7)

val mlpKernel = new MLPKernel(1.5, 0.75)

val whiteNoiseKernel = new DiracKernel(0.2)
whiteNoiseKernel.block_all_hyper_parameters

OmniOSA.gridSize = 4
OmniOSA.gridStep = 0.15
OmniOSA.globalOpt = "ML"
OmniOSA.maxIterations = 250

OmniOSA.clearExogenousVars()
//Get test results for Linear GP-AR model
//with a mean function given by the persistence
//model
OmniOSA.setTarget(40, 10)
val resPolyAR = OmniOSA.buildAndTestGP(
  mlpKernel+tKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)


OmniOSA.clearExogenousVars()
//Set solar wind speed and IMF Bz as exogenous variables
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 16), List(1,3))
//Reset kernel and noise to initial states
mlpKernel.setw(1.2901485870065708)
mlpKernel.setoffset(73.92009461973996)
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0
//Get test results for a GP-ARX model
//with a mean function given by the persistence
//model
val resPolyARX = OmniOSA.buildAndTestGP(
  mlpKernel + tKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)


//Compare with base line of the Persistence model
val resPer = DstPersistenceMOExperiment(0)

OmniOSA.clearExogenousVars()

mlpKernel.setw(80.0)
mlpKernel.setoffset(20.0)
rbfKernel.setbandwidth(1.7)
OmniOSA.gridSize = 3
OmniOSA.globalOpt = "ML"
OmniOSA.modelType_("GP-NARMAX")
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 15, 16, 28), List(4), changeModelType = false)

val resPolyNM = OmniOSA.buildAndTestGP(
  tKernel+mlpKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)

//Print the results out on the console
resPer.print()
resPolyAR.print()
resPolyARX.print()
resPolyNM.print()

OmniOSA.clearExogenousVars()

OmniOSA.globalOpt = "CSA"
OmniOSA.gridSize = 4
OmniOSA.maxIterations = 30
OmniOSA.useLogScale = true

mlpKernel.setw(10.0)
mlpKernel.setoffset(10.0)
OmniOSA.modelType_("GP-AR")

OmniOSA.experiment(
  tKernel+mlpKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence,
  3 to 12)


mlpKernel.setw(10.0)
mlpKernel.setoffset(10.0)
OmniOSA.modelType_("GP-ARX")

OmniOSA.experiment(
  tKernel+mlpKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence,
  8 to 12)

//Use best performing model for generating predictions
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0
OmniOSA.globalOpt = "GS"
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 16), List(1,3))
mlpKernel.setw(1.2901485870065708)
mlpKernel.setoffset(73.92009461973996)
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0

val predictPipeline =
  OmniOSA.dataPipeline > OmniOSA.gpTrain(
      tKernel+mlpKernel,
      whiteNoiseKernel,
      OmniOSA.meanFuncPersistence) > OmniOSA.generateGPPredictions(hoursOffset = 40)

predictPipeline(OmniOSA.trainingDataSections)


OmniOSA.globalOpt = "CSA"
OmniOSA.setTarget(40, 7)
OmniOSA.setExogenousVars(List(24, 16), List(1,3))

val resSGPARX = OmniOSA.buildAndTestSGP(
  mlpKernel + tKernel, whiteNoiseKernel, -4.4, -0.01,
  OmniOSA.meanFuncPersistence)

//val mlpKernel = new MLPKernel(2.5, 0.9)

//val whiteNoiseKernel = new DiracKernel(0.5)

val sgpPipeline =
  OmniOSA.dataPipeline > OmniOSA.sgpTrain(
    mlpKernel + tKernel,
    whiteNoiseKernel, -4.4, -0.01,
    OmniOSA.meanFuncPersistence) > OmniOSA.generateSGPPredictions()

sgpPipeline(OmniOSA.trainingDataSections)



OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0
OmniOSA.globalOpt = "GS"
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 16), List(1,3))
mlpKernel.setw(1.2901485870065708)
mlpKernel.setoffset(73.92009461973996)
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0


val modelAndScales = (
  OmniOSA.dataPipeline >
    OmniOSA.gpTrain(
      tKernel+mlpKernel, whiteNoiseKernel,
      OmniOSA.meanFuncPersistence))(
  OmniOSA.trainingDataSections)


val pipeline_alt = {
  fileToStream > replaceWhiteSpaces > StreamDataPipe(OmniOSA.getStormTimeRanges() > OmniOSA.processTimeSegment)
}

val pipeline2 = {
  DataPipe((testData: OmniOSA.Data) =>
    modelAndScales._1
      .test(testData).map(t => (DenseVector(t._3), DenseVector(t._2)))
      .toStream) >
    StreamDataPipe((c: (OmniOSA.Features, OmniOSA.Features)) => (c._1(0), c._2(0))) >
    DataPipe((results: Stream[(Double, Double)]) =>
      new RegressionMetrics(results.toList, results.length)
        .setName(OmniOSA._modelType+" "+OmniOSA.columnNames(OmniOSA._targetColumn)+"; OSA")
    )
}

val stormAvgPipeline = pipeline_alt > StreamDataPipe(pipeline2)

val mets = stormAvgPipeline.run(OmniOSA.dataDir+OmniOSA.stormsFileJi)

val perfs = mets.map(r => (r.rmse, r.mae, r.corr))
val rmse = perfs.map(_._1).sum/perfs.length.toDouble

OmniOSA.clearExogenousVars()

OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0
OmniOSA.globalOpt = "GS"
OmniOSA.setTarget(40, 6)

mlpKernel.setw(1.2901485870065708)
mlpKernel.setoffset(73.92009461973996)
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0


13.25
10.71