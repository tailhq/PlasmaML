package io.github.mandar2812.PlasmaML.omni

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian
import io.github.mandar2812.PlasmaML.omni.OmniMSA.Features
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.PlasmaML.omni.OmniMultiOutputModels._
import io.github.mandar2812.PlasmaML.omni.OmniOSA.DataAndScales
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.modelpipe.ModelPredictionPipe
import io.github.mandar2812.dynaml.models.stp.MVStudentsTModel
import io.github.mandar2812.dynaml.optimization.{AbstractCSA, CoupledSimulatedAnnealing, FFBackProp, GridSearch}
import io.github.mandar2812.dynaml.probability.RandomVariable
import org.apache.log4j.Logger

/**
  * Contains helper methods to train models
  * for Multiple Step Ahead (multiple hour ahead)
  * prediction of OMNI time series.
  *
  * @author mandar2812 date 05/04/2017.
  * */
object OmniMSA {

  /*
  * Instantiating some types to make code more
  * readable later on.
  * */
  type Features = DenseVector[Double]
  type Targets = DenseVector[Double]
  type Data = Iterable[(Features, Targets)]
  type ScoresAndLabels = Iterable[(Targets, Targets)]
  type DataScales = (GaussianScaler, GaussianScaler)
  type LayerParams = (DenseMatrix[Double], DenseVector[Double])


  var quietTimeSegment = ("2014/01/10/00", "2014/12/30/00")

  def haarWaveletPipe = IterableDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
    if (useWaveletBasis)
      (gHFeat(featAndTarg._1), gHTarg(featAndTarg._2))
    else
      featAndTarg
  )


  def prepareData: DataPipe[String, (Data, DataScales)] = {

    val (pF, pT) = (math.pow(2, orderFeat).toInt,math.pow(2, orderTarget).toInt)

    val prepareTrainingData = fileToStream >
      replaceWhiteSpaces >
      IterableDataPipe((stormEventData: String) => {
        val stormMetaFields = stormEventData.split(',')

        val startDate = stormMetaFields(1)
        val startHour = stormMetaFields(2).take(2)

        val endDate = stormMetaFields(3)
        val endHour = stormMetaFields(4).take(2)

        (startDate+"/"+startHour, endDate+"/"+endHour)
      }) >
      DataPipe((s: Iterable[(String, String)]) =>
        s ++ Stream(quietTimeSegment)) >
      IterableFlatMapPipe((storm: (String, String)) => {
        // for each storm construct a data set

        val (trainingStartDate, trainingEndDate) =
          (formatter.parseDateTime(storm._1).minusHours(pF),
            formatter.parseDateTime(storm._2).plusHours(pT))

        val (trStampStart, trStampEnd) =
          (trainingStartDate.getMillis/1000.0, trainingEndDate.getMillis/1000.0)

        val filterData = IterableDataPipe((couple: (Double, DenseVector[Double])) =>
          couple._1 >= trStampStart && couple._1 <= trStampEnd)

        val generateDataSegment = preProcess >
          extractTimeSeriesVec((year,day,hour) => {
            val dt = dayofYearformat.parseDateTime(
              year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
            dt.getMillis/1000.0
          }) >
          filterData >
          OmniMultiOutputModels.deltaOperationARXMult(List.fill(1+exogenousInputs.length)(pF), pT)

        generateDataSegment("data/omni2_"+trainingStartDate.getYear+".csv")
      })

    prepareTrainingData >
      haarWaveletPipe >
      gaussianScaling
  }

  /**
    * Trains a [[GenericFFNeuralNet]] on the storms
    * contained in [[OmniOSA.stormsFile3]] plus one year of
    * OMNI data from the year 2014.
    *
    * @param alpha The learning rate for [[FFBackProp]]
    * @param reg The regularisation parameter
    * @param momentum Momentum parameter
    * @param maxIt Maximum number of iterations to run for [[FFBackProp]]
    * @param useWaveletBasis Set to true if discrete wavelet transform is
    *                        to be used for pre-processing data.
    *
    * */
  def train(
    alpha: Double = 0.01, reg: Double = 0.001,
    momentum: Double = 0.02, maxIt: Int = 20,
    mini: Double = 1.0, useWaveletBasis: Boolean = true):
  (GenericFFNeuralNet[Data, LayerParams, Features], DataScales) = {

    val modelTrain = DataPipe((dataAndScales: (Data, DataScales)) => {

      val (num_features, num_targets) = (dataAndScales._1.head._1.length, dataAndScales._1.head._2.length)

      val layerCounts = List(num_features) ++ neuronCounts ++ List(num_targets)

      val stackFactory = NeuralStackFactory(layerCounts)(activations)

      val uni = Gaussian(0.0, 1.0)

      val weightsInitializer = RandomVariable(
        layerCounts.sliding(2)
          .toSeq
          .map(l => (l.head, l.last))
          .map((c) => RandomVariable(() => (
            DenseMatrix.tabulate(c._2, c._1)((_, _) => uni.draw()),
            DenseVector.tabulate(c._2)(_ => uni.draw())))
          ):_*
      )

      val backPropOptimizer =
        new FFBackProp(stackFactory)
          .setNumIterations(maxIt)
          .setRegParam(reg)
          .setStepSize(alpha)
          .setMiniBatchFraction(mini)
          .momentum_(momentum)

      val model = GenericFFNeuralNet(
        backPropOptimizer,
        dataAndScales._1, DataPipe[Data, Stream[(Features, Targets)]](_.toStream),
        weightsInitializer)

      model.learn()

      (model, dataAndScales._2)
    })

    val netWorkFlow = prepareData > modelTrain

    netWorkFlow(OmniOSA.dataDir+OmniOSA.stormsFile3)
  }

  def train(
    kernel: LocalScalarKernel[Features], noise: LocalScalarKernel[Features],
    gridSize: Int, gridStep: Double, logSc: Boolean,
    globalOpt: String, maxIt: Int,
    phi: DataPipe[Features, DenseVector[Double]]):
  (MVStudentsTModel[Data, Features] , DataScales) = {

    implicit val transform = DataPipe[Data, Seq[(Features, Targets)]]((d: Data) => d.toSeq)

    val modelTrain = DataPipe((dataAndScales: (Data, DataScales)) => {
      val multiVariateSTModel = MVStudentsTModel[Data, Features](kernel, noise, phi) _

      val num_outputs = dataAndScales._2._2.mean.length
      val initial_model = multiVariateSTModel(dataAndScales._1, dataAndScales._1.toSeq.length, num_outputs)

      val gs = globalOpt match {

        case "CSA" =>
          new CoupledSimulatedAnnealing[initial_model.type](initial_model)
            .setGridSize(gridSize)
            .setStepSize(gridStep)
            .setLogScale(logSc)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
        case _ =>
          new GridSearch[initial_model.type](initial_model)
            .setGridSize(gridSize)
            .setStepSize(gridStep)
            .setLogScale(logSc)
      }

      val startConf = initial_model.covariance.effective_state ++ initial_model.noiseModel.effective_state

      val (optModel, _) = gs.optimize(startConf, Map("persist" -> "true"))

      (optModel, dataAndScales._2)
    })

    val netWorkFlow = prepareData > modelTrain

    netWorkFlow(OmniOSA.dataDir+OmniOSA.stormsFile3)
  }

  def test(model: MVStudentsTModel[Data, Features], scaler: DataScales) = {

    val (pF, pT) = (math.pow(2, orderFeat).toInt,math.pow(2, orderTarget).toInt)

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val testDataFilter = IterableDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val scaleFeatures: DataPipe[Features, Features] =
      if (useWaveletBasis) gHFeat > scaler._1
      else scaler._1

    val scaleTarg: DataPipe[Features, Features] =
      if (useWaveletBasis) gHTarg > scaler._2
      else scaler._2

    val reverseScale: DataPipe[Features, Features] =
      if (useWaveletBasis) scaler._2.i > gHTarg.i
      else scaler._2.i

    val finalPipe = preProcess >
      extractTimeSeriesVec((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
        dt.getMillis/1000.0
      }) >
      testDataFilter >
      OmniMultiOutputModels.deltaOperationARXMult(List.fill(1+exogenousInputs.length)(pF), pT) >
      DataPipe((data: Data) => {
        val forSc = scaleFeatures*scaleTarg
        val revSc = reverseScale*reverseScale
        revSc(model.test(forSc(data)).toStream.map(t => (t._3, t._2)))
      }) >
      DataPipe((predictions: ScoresAndLabels) => {
        val sc = predictions.toList
        val metrics = new MultiRegressionMetrics(
          sc,
          sc.length)
        metrics.setName(names(column)+" "+pT+" hour forecast")
      })

    finalPipe("data/omni2_"+testStartDate.getYear+".csv")


  }

  /**
    * Test a trained [[GenericFFNeuralNet]] on the time period
    * between [[OmniMultiOutputModels.testStart]] and [[OmniMultiOutputModels.testEnd]]
    *
    * @param model The model to be tested
    * @param scaler The [[GaussianScaler]] objects returned by [[train()]].
    * */
  def test(
    model: GenericFFNeuralNet[Data, LayerParams, Features],
    scaler: DataScales): MultiRegressionMetrics = {

    val (pF, pT) = (math.pow(2, orderFeat).toInt,math.pow(2, orderTarget).toInt)

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val testDataFilter = IterableDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val scaleData: DataPipe[Features, Features] =
      if (useWaveletBasis) gHFeat > scaler._1
      else scaler._1

    val reverseScale: DataPipe[Features, Features] =
      if (useWaveletBasis) scaler._2.i > gHTarg.i
      else scaler._2.i

    val modelPredict = ModelPredictionPipe[
      Data, Features, Features, Targets,
      Targets, GenericFFNeuralNet[Data, LayerParams, Features]](
      scaleData, model, reverseScale)

    val finalPipe = preProcess >
      extractTimeSeriesVec((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
        dt.getMillis/1000.0
      }) >
      testDataFilter >
      OmniMultiOutputModels.deltaOperationARXMult(List.fill(1+exogenousInputs.length)(pF), pT) >
      IterableDataPipe(modelPredict*identityPipe[Targets]) >
      DataPipe((predictions: ScoresAndLabels) => {
        val sc = predictions.toList
        val metrics = new MultiRegressionMetrics(
          sc,
          sc.length)
        metrics.setName(names(column)+" "+pT+" hour forecast")
      })

    finalPipe("data/omni2_"+testStartDate.getYear+".csv")

  }

  def test(): MultiRegressionMetrics = {

    val (pF, pT) = (1,math.pow(2, orderTarget).toInt)

    OmniMultiOutputModels.exogenousInputs = List()

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val filterData = IterableDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val prepareFeaturesAndOutputs = extractTimeSeries((year,day,hour) => {
      val dt = dayofYearformat.parseDateTime(
        year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString)
      dt.getMillis/1000.0 }) >
      filterData >
      deltaOperationMult(pF, pT)

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      IterableDataPipe((c: (Features, Targets)) =>
        (DenseVector.fill[Double](c._2.length)(c._1(c._1.length-1)), c._2)) >
      DataPipe((d: Iterable[(DenseVector[Double], DenseVector[Double])]) => {
        val sc = d.toList
        new MultiRegressionMetrics(sc, sc.length)
      })

    flow("data/omni2_"+testStartDate.getYear+".csv")


  }

  /**
    * Train and test a [[GenericFFNeuralNet]] model on the OMNI data.
    * Training set is the period between
    * [[OmniMultiOutputModels.trainingStart]] to [[OmniMultiOutputModels.trainingEnd]]
    *
    * Test set is the time interval between
    * [[OmniMultiOutputModels.testStart]] to [[OmniMultiOutputModels.testEnd]]
    *
    * @param alpha Learning rate
    * @param reg Regularization parameter
    * @param momentum Momentum parameter
    * @param maxIt The maximum iterations of [[FFBackProp]] to run
    * @param mini Mini batch fraction to be used in each epoch of [[FFBackProp]]
    *
    * @return An instance of [[MultiRegressionMetrics]] containing the test results.
    * */
  def apply(
    alpha: Double = 0.01, reg: Double = 0.001,
    momentum: Double = 0.02, maxIt: Int = 20,
    mini: Double = 1.0): MultiRegressionMetrics = {


    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val (hFeat, hTarg) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val (trainingStartDate, trainingEndDate) =
      (formatter.parseDateTime(trainingStart).minusHours(pF+pT),
        formatter.parseDateTime(trainingEnd))

    val (trStampStart, trStampEnd) =
      (trainingStartDate.getMillis/1000.0, trainingEndDate.getMillis/1000.0)

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF+pT),
        formatter.parseDateTime(testEnd))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val trainPipe = IterableDataPipe((couple: (Double, Double)) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val testPipe = IterableDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val postPipe = deltaOperationMult(pF,pT)

    val haarWaveletPipe = IterableDataPipe((featAndTarg: (Features, Targets)) =>
      (hFeat*hTarg)(featAndTarg))

    val modelTrainTestWavelet =
      DataPipe((trainTest: (Data, Data, DataScales)) => {

        val reverseTargetsScaler = trainTest._3._2.i * trainTest._3._2.i

        val layerCounts = List(trainTest._1.head._1.length) ++ neuronCounts ++ List(trainTest._1.head._2.length)

        val stackFactory = NeuralStackFactory(layerCounts)(activations)

        val weightsInitializer = GenericFFNeuralNet.getWeightInitializer(layerCounts)

        val backPropOptimizer =
          new FFBackProp(stackFactory)
            .setNumIterations(maxIt)
            .setRegParam(reg)
            .setStepSize(alpha)
            .setMiniBatchFraction(mini)
            .momentum_(momentum)

        val model = GenericFFNeuralNet(
          backPropOptimizer,
          trainTest._1, DataPipe[Data, Stream[(Features, Targets)]](_.toStream),
          weightsInitializer)


        model.learn()

        val testSetToResult = DataPipe(
          (testSet: Iterable[(Features, Targets)]) => testSet.map(c => (model.predict(c._1), c._2))) >
          IterableDataPipe(
            (tuple: (Targets, Targets)) => reverseTargetsScaler(tuple)) >
          IterableDataPipe(
            (tuple: (Targets, Targets)) => (hTarg.i*hTarg.i)(tuple)) >
          DataPipe((scoresAndLabels: Iterable[(Targets, Targets)]) => {
            val sc = scoresAndLabels.toList
            val metrics = new MultiRegressionMetrics(
              sc,
              sc.length)
            metrics.setName(names(column)+" "+pT+" hour forecast")
          })

        testSetToResult(trainTest._2)
      })

    val modelTrainTest = DataPipe((trainTest: (Data, Data, DataScales)) => {

      val reverseTargetsScaler = trainTest._3._2.i * trainTest._3._2.i

      val layerCounts = List(trainTest._1.head._1.length) ++ neuronCounts ++ List(trainTest._1.head._2.length)

      val stackFactory = NeuralStackFactory(layerCounts)(activations)

      val weightsInitializer = GenericFFNeuralNet.getWeightInitializer(layerCounts)

      val backPropOptimizer =
        new FFBackProp(stackFactory)
          .setNumIterations(maxIt)
          .setRegParam(reg)
          .setStepSize(alpha)
          .setMiniBatchFraction(mini)
          .momentum_(momentum)

      val model = GenericFFNeuralNet(
        backPropOptimizer,
        trainTest._1, DataPipe[Data, Stream[(Features, Targets)]](_.toStream),
        weightsInitializer)


      model.learn()

      val testSetToResult = DataPipe((testSet: Data) => testSet.map(c => (model.predict(c._1), c._2))) >
        IterableDataPipe((tuple: (Targets, Targets)) =>
          reverseTargetsScaler(tuple)) >
        DataPipe((predictions: ScoresAndLabels) => {
          val sc = predictions.toList
          val metrics = new MultiRegressionMetrics(
            sc,
            sc.length)
          metrics.setName(names(column)+" "+pT+" hour forecast")
        })

      testSetToResult(trainTest._2)
    })


    val finalPipe = if(useWaveletBasis) {
      duplicate(preProcess >
        extractTimeSeries((year,day,hour) => {
          val dt = dayofYearformat.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
          dt.getMillis/1000.0 })) >
        DataPipe(trainPipe, testPipe) >
        duplicate(postPipe > haarWaveletPipe) >
        gaussianScalingTrainTest >
        modelTrainTestWavelet
    } else {
      duplicate(preProcess >
        extractTimeSeries((year,day,hour) => {
          val dt = dayofYearformat.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
          dt.getMillis/1000.0 })) >
        DataPipe(trainPipe, testPipe) >
        duplicate(postPipe) >
        gaussianScalingTrainTest >
        modelTrainTest
    }

    finalPipe((
      "data/omni2_"+trainingStartDate.getYear+".csv",
      "data/omni2_"+testStartDate.getYear+".csv"))

  }

}


object DstMSAExperiment {

  val logger = Logger.getLogger(this.getClass)

  var learningRate: Double = 1.0

  var reg: Double = 0.0005

  var momentum: Double = 0.6

  var it:Int = 150

  var gridSize = 3
  var gridStep = 0.5
  var useLogScale = false

  var globalOpt = "GS"

  /**
    * Train and test a [[GenericFFNeuralNet]] model on the OMNI data.
    * Training set is combination of 20 storms from [[OmniOSA.stormsFile2]]
    * and one year period of quite time.
    *
    * Test set is the storm events in [[OmniOSA.stormsFileJi]]
    *
    * @param orderF The base 2 logarithm of the auto-regressive order for each input
    * @param orderT The base 2 logarithm of the auto-regressive order for the outputs
    * @param useWavelets Set to true if you want to use discrete wavelet transform to
    *                    pre-process input and output features before training.
    *
    * @return An instance of [[MultiRegressionMetrics]] containing the test results.
    * */
  def apply(orderF: Int = 4, orderT: Int = 3, useWavelets: Boolean = true) = {

    OmniMultiOutputModels.orderFeat = orderF
    OmniMultiOutputModels.orderTarget = orderT
    OmniMultiOutputModels.useWaveletBasis = useWavelets

    val (model, scaler) = OmniMSA.train(learningRate, reg, momentum, it, 1.0)

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        DataPipe((st: Iterable[String]) => st.take(43)) >
        IterableDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          OmniMultiOutputModels.testStart = startDate+"/"+startHour
          OmniMultiOutputModels.testEnd = endDate+"/"+endHour

          logger.info("Testing on Storm: "+OmniMultiOutputModels.testStart+" to "+OmniMultiOutputModels.testEnd)

          OmniMSA.test(model, scaler)
        }) >
        DataPipe((metrics: Iterable[MultiRegressionMetrics]) =>
          metrics.reduce((m,n) => m++n))

    stormsPipe("data/geomagnetic_storms.csv")

  }

  def apply(
    kernel: LocalScalarKernel[Features],
    noise: LocalScalarKernel[Features],
    orderF: Int, orderT: Int,
    useWavelets: Boolean) = {


    OmniMultiOutputModels.orderFeat = orderF
    OmniMultiOutputModels.orderTarget = orderT
    OmniMultiOutputModels.useWaveletBasis = useWavelets

    val (model, scaler) = OmniMSA.train(
      kernel, noise,
      gridSize, gridStep,
      useLogScale, globalOpt, it,
      DataPipe((x: Features) => DenseVector(x.toArray :+ 1.0))
    )

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        DataPipe((st: Iterable[String]) => st.take(43)) >
        IterableDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          OmniMultiOutputModels.testStart = startDate+"/"+startHour
          OmniMultiOutputModels.testEnd = endDate+"/"+endHour

          logger.info("Testing on Storm: "+OmniMultiOutputModels.testStart+" to "+OmniMultiOutputModels.testEnd)

          OmniMSA.test(model, scaler)
        }) >
        DataPipe((metrics: Iterable[MultiRegressionMetrics]) =>
          metrics.reduce((m,n) => m++n))

    stormsPipe("data/geomagnetic_storms.csv")

  }
}
