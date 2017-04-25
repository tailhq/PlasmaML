package io.github.mandar2812.PlasmaML.omni

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.PlasmaML.omni.OmniMultiOutputModels._
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import io.github.mandar2812.dynaml.modelpipe.ModelPredictionPipe
import io.github.mandar2812.dynaml.optimization.FFBackProp
import org.apache.log4j.Logger

/**
  * Contains helper methods to train Neural Nets
  * for Multiple Step Ahead (multiple hour ahead)
  * prediction of OMNI time series.
  *
  * @author mandar2812 date 05/04/2017.
  * */
object OmniMSANN {

  /*
  * Instantiating some types to make code more
  * readable later on.
  * */
  type Features = DenseVector[Double]
  type Targets = DenseVector[Double]
  type Data = Stream[(Features, Targets)]
  type ScoresAndLabels = Stream[(Targets, Targets)]
  type DataScales = (GaussianScaler, GaussianScaler)
  type LayerParams = (DenseMatrix[Double], DenseVector[Double])


  def haarWaveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
    if (useWaveletBasis)
      (gHFeat(featAndTarg._1), gHTarg(featAndTarg._2))
    else
      featAndTarg
  )

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

    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)

    val prepareTrainingData = fileToStream >
      replaceWhiteSpaces >
      StreamDataPipe((stormEventData: String) => {
        val stormMetaFields = stormEventData.split(',')

        val startDate = stormMetaFields(1)
        val startHour = stormMetaFields(2).take(2)

        val endDate = stormMetaFields(3)
        val endHour = stormMetaFields(4).take(2)

        (startDate+"/"+startHour, endDate+"/"+endHour)
      }) >
      DataPipe((s: Stream[(String, String)]) =>
        s ++ Stream(("2014/01/10/00", "2014/12/30/00"))) >
      StreamFlatMapPipe((storm: (String, String)) => {
        // for each storm construct a data set

        val (trainingStartDate, trainingEndDate) =
          (formatter.parseDateTime(storm._1).minusHours(pF),
            formatter.parseDateTime(storm._2).plusHours(pT))

        val (trStampStart, trStampEnd) =
          (trainingStartDate.getMillis/1000.0, trainingEndDate.getMillis/1000.0)

        val filterData = StreamDataPipe((couple: (Double, DenseVector[Double])) =>
          couple._1 >= trStampStart && couple._1 <= trStampEnd)

        val generateDataSegment = preProcess >
          extractTimeSeriesVec((year,day,hour) => {
            val dt = dayofYearformat.parseDateTime(
              year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
            dt.getMillis/1000.0
          }) >
          filterData

        generateDataSegment("data/omni2_"+trainingStartDate.getYear+".csv")
      })

    val modelTrain = DataPipe((dataAndScales: (Data, DataScales)) => {

      val (num_features, num_targets) = (dataAndScales._1.head._1.length, dataAndScales._1.head._2.length)

      val layerCounts = List(num_features) ++ neuronCounts ++ List(num_targets)

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
        dataAndScales._1, identityPipe[Data],
        weightsInitializer)

      model.learn()

      (model, dataAndScales._2)
    })

    val netWorkFlow = prepareTrainingData >
      OmniMultiOutputModels.deltaOperationARXMult(List.fill(1+exogenousInputs.length)(pF), pT) >
      haarWaveletPipe >
      gaussianScaling >
      modelTrain

    netWorkFlow(OmniOSA.dataDir+OmniOSA.stormsFile3)
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

    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val testDataFilter = StreamDataPipe((couple: (Double, DenseVector[Double])) =>
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
      StreamDataPipe(modelPredict*identityPipe[Targets]) >
      DataPipe((predictions: ScoresAndLabels) => {
        val metrics = new MultiRegressionMetrics(
          predictions.toList,
          predictions.length)
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

    val filterData = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val prepareFeaturesAndOutputs = extractTimeSeries((year,day,hour) => {
      val dt = dayofYearformat.parseDateTime(
        year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString)
      dt.getMillis/1000.0 }) >
      filterData >
      deltaOperationMult(pF, pT)

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      StreamDataPipe((c: (Features, Targets)) =>
        (DenseVector.fill[Double](c._2.length)(c._1(c._1.length-1)), c._2)) >
      DataPipe((d: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        new MultiRegressionMetrics(d.toList, d.length)
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

    val trainPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val testPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val postPipe = deltaOperationMult(pF,pT)

    val haarWaveletPipe = StreamDataPipe((featAndTarg: (Features, Targets)) =>
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
          trainTest._1, identityPipe[Stream[(Features, Targets)]],
          weightsInitializer)


        model.learn()

        val testSetToResult = DataPipe(
          (testSet: Stream[(Features, Targets)]) => testSet.map(c => (model.predict(c._1), c._2))) >
          StreamDataPipe(
            (tuple: (Targets, Targets)) => reverseTargetsScaler(tuple)) >
          StreamDataPipe(
            (tuple: (Targets, Targets)) => (hTarg.i*hTarg.i)(tuple)) >
          DataPipe((scoresAndLabels: Stream[(Targets, Targets)]) => {
            val metrics = new MultiRegressionMetrics(
              scoresAndLabels.toList,
              scoresAndLabels.length)
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
        trainTest._1, identityPipe[Data],
        weightsInitializer)


      model.learn()

      val testSetToResult = DataPipe((testSet: Data) => testSet.map(c => (model.predict(c._1), c._2))) >
        StreamDataPipe((tuple: (Targets, Targets)) =>
          reverseTargetsScaler(tuple)) >
        DataPipe((predictions: ScoresAndLabels) => {
          val metrics = new MultiRegressionMetrics(
            predictions.toList,
            predictions.length)
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


object DstMSANNExperiment {

  val logger = Logger.getLogger(this.getClass)

  var learningRate: Double = 1.0

  var reg: Double = 0.0005

  var momentum: Double = 0.6

  var it:Int = 150

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

    val (model, scaler) = OmniMSANN.train(learningRate, reg, momentum, it, 1.0)

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        DataPipe((st: Stream[String]) => st.take(43)) >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          OmniMultiOutputModels.testStart = startDate+"/"+startHour
          OmniMultiOutputModels.testEnd = endDate+"/"+endHour

          logger.info("Testing on Storm: "+OmniMultiOutputModels.testStart+" to "+OmniMultiOutputModels.testEnd)

          OmniMSANN.test(model, scaler)
        }) >
        DataPipe((metrics: Stream[MultiRegressionMetrics]) =>
          metrics.reduce((m,n) => m++n))

    stormsPipe("data/geomagnetic_storms.csv")

  }
}
