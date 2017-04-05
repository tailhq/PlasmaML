package io.github.mandar2812.PlasmaML.omni

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.neuralnets.{FeedForwardNetwork, GenericFFNeuralNet, NeuralStackFactory}
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.PlasmaML.omni.OmniMultiOutputModels._
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.optimization.FFBackProp
import org.apache.log4j.Logger

/**
  * Created by mandar on 05/04/2017.
  */
object OmniMSANN {

  type Features = DenseVector[Double]
  type Data = Stream[(Features, Features)]
  type LayerParams = (DenseMatrix[Double], DenseVector[Double])

  def train(alpha: Double = 0.01, reg: Double = 0.001,
            momentum: Double = 0.02, maxIt: Int = 20,
            mini: Double = 1.0, useWaveletBasis: Boolean = true)
  : (GenericFFNeuralNet[Data, LayerParams, Features],
    (GaussianScaler, GaussianScaler)) = {

    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val (hFeat, hTarg) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val haarWaveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
      if (useWaveletBasis) (hFeat*hTarg)(featAndTarg) else featAndTarg)


    val prepareTrainingData = DataPipe((n: Int) => {

      val stormsPipe =
        fileToStream >
          replaceWhiteSpaces >
          StreamDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            (startDate+"/"+startHour,
              endDate+"/"+endHour)
          }) >
          DataPipe((s: Stream[(String, String)]) =>
            s.takeRight(n) ++ Stream(("2014/11/15/00", "2014/12/15/00"))) >
          StreamDataPipe((storm: (String, String)) => {
            // for each storm construct a data set

            val (trainingStartDate, trainingEndDate) =
              (formatter.parseDateTime(storm._1).minusHours(pF),
                formatter.parseDateTime(storm._2).plusHours(pT))

            val (trStampStart, trStampEnd) =
              (trainingStartDate.getMillis/1000.0, trainingEndDate.getMillis/1000.0)

            val filterTrainingData = StreamDataPipe((couple: (Double, Double)) =>
              couple._1 >= trStampStart && couple._1 <= trStampEnd)

            val getTraining = preProcess >
              extractTimeSeries((year,day,hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
                dt.getMillis/1000.0
              }) >
              filterTrainingData >
              deltaOperationMult(pF,pT) >
              haarWaveletPipe

            getTraining("data/omni2_"+trainingStartDate.getYear+".csv")

          }) >
          DataPipe((s: Stream[Stream[(DenseVector[Double], DenseVector[Double])]]) => {
            s.reduce((p,q) => p ++ q)
          })

      stormsPipe("data/geomagnetic_storms.csv")
    })

    val modelTrain = (trainTest: (Stream[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler))) => {


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
        trainTest._1, identityPipe[Stream[(DenseVector[Double], DenseVector[Double])]],
        weightsInitializer)


      model.learn()

      (model, trainTest._2)
    }

    (prepareTrainingData >
      gaussianScaling >
      DataPipe(modelTrain)) run 20

  }

  def test(model: GenericFFNeuralNet[Data, LayerParams, Features],
           scaler: (GaussianScaler, GaussianScaler)): MultiRegressionMetrics = {

    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val (hFeat, hTarg) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val testPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val postPipe = deltaOperationMult(pF,pT)

    val haarWaveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
      if (useWaveletBasis) (hFeat*hTarg)(featAndTarg) else featAndTarg)

    val reverseTargetsScaler = scaler._2.i * scaler._2.i

    val modelTest = DataPipe((testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => {
      val testSetToResult = DataPipe(
        (testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => testSet.map(c => (model.predict(c._1), c._2))) >
        StreamDataPipe(
          (tuple: (DenseVector[Double], DenseVector[Double])) => reverseTargetsScaler(tuple)) >
        StreamDataPipe(
          (tuple: (DenseVector[Double], DenseVector[Double])) =>
            if (useWaveletBasis) (hTarg.i*hTarg.i)(tuple) else tuple) >
        DataPipe((scoresAndLabels: Stream[(DenseVector[Double], DenseVector[Double])]) => {
          val metrics = new MultiRegressionMetrics(
            scoresAndLabels.toList,
            scoresAndLabels.length)
          metrics.setName(names(column)+" "+pT+" hour forecast")
        })

      testSetToResult(testSet)

    })

    val finalPipe = preProcess >
      extractTimeSeries((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
        dt.getMillis/1000.0
      }) >
      testPipe >
      postPipe >
      haarWaveletPipe >
      StreamDataPipe((tuple: (DenseVector[Double], DenseVector[Double])) => (scaler._1*scaler._2)(tuple)) >
      modelTest

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
      StreamDataPipe((c: (DenseVector[Double], DenseVector[Double])) =>
        (DenseVector.fill[Double](c._2.length)(c._1(c._1.length-1)), c._2)) >
      DataPipe((d: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        new MultiRegressionMetrics(d.toList, d.length)
      })

    flow("data/omni2_"+testStartDate.getYear+".csv")


  }

  def apply(alpha: Double = 0.01, reg: Double = 0.001,
            momentum: Double = 0.02, maxIt: Int = 20,
            mini: Double = 1.0): MultiRegressionMetrics = {


    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val (hFeat, hTarg) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))



    val (trainingStartDate, trainingEndDate) =
      (formatter.parseDateTime(trainingStart).minusHours(pF+pT),
        formatter.parseDateTime(trainingEnd))

    val (trStampStart, trStampEnd) =
      (trainingStartDate.getMillis/1000.0, trainingEndDate.getMillis/1000.0)

    val (validationStartDate, validationEndDate) =
      (formatter.parseDateTime(validationStart).minusHours(pF+pT),
        formatter.parseDateTime(validationEnd))

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF+pT),
        formatter.parseDateTime(testEnd))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val trainPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val testPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val postPipe = deltaOperationMult(pF,pT)

    val haarWaveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
      (hFeat*hTarg)(featAndTarg))

    val modelTrainTestWavelet =
      DataPipe((trainTest:
                (Stream[(DenseVector[Double], DenseVector[Double])],
                  Stream[(DenseVector[Double], DenseVector[Double])],
                  (GaussianScaler, GaussianScaler))) => {

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
          trainTest._1, identityPipe[Stream[(DenseVector[Double], DenseVector[Double])]],
          weightsInitializer)


        model.learn()

        val testSetToResult = DataPipe(
          (testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => testSet.map(c => (model.predict(c._1), c._2))) >
          StreamDataPipe(
            (tuple: (DenseVector[Double], DenseVector[Double])) => reverseTargetsScaler(tuple)) >
          StreamDataPipe(
            (tuple: (DenseVector[Double], DenseVector[Double])) => (hTarg.i*hTarg.i)(tuple)) >
          DataPipe((scoresAndLabels: Stream[(DenseVector[Double], DenseVector[Double])]) => {
            val metrics = new MultiRegressionMetrics(
              scoresAndLabels.toList,
              scoresAndLabels.length)
            metrics.setName(names(column)+" "+pT+" hour forecast")
          })

        testSetToResult(trainTest._2)
      })

    val modelTrainTest =
      DataPipe((trainTest:
                (Stream[(DenseVector[Double], DenseVector[Double])],
                  Stream[(DenseVector[Double], DenseVector[Double])],
                  (GaussianScaler, GaussianScaler))) => {


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
          trainTest._1, identityPipe[Stream[(DenseVector[Double], DenseVector[Double])]],
          weightsInitializer)


        model.learn()

        val testSetToResult = DataPipe(
          (testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => testSet.map(c => (model.predict(c._1), c._2))) >
          StreamDataPipe(
            (tuple: (DenseVector[Double], DenseVector[Double])) => reverseTargetsScaler(tuple)) >
          DataPipe((scoresAndLabels: Stream[(DenseVector[Double], DenseVector[Double])]) => {
            val metrics = new MultiRegressionMetrics(
              scoresAndLabels.toList,
              scoresAndLabels.length)
            metrics.setName(names(column)+" "+pT+" hour forecast")
          })

        testSetToResult(trainTest._2)
      })


    val finalPipe = useWaveletBasis match {
      case true =>
        duplicate(preProcess >
          extractTimeSeries((year,day,hour) => {
            val dt = dayofYearformat.parseDateTime(
              year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
            dt.getMillis/1000.0 })) >
          DataPipe(trainPipe, testPipe) >
          duplicate(postPipe > haarWaveletPipe) >
          gaussianScalingTrainTest >
          modelTrainTestWavelet

      case false =>
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

          val eventId = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          //val minDst = stormMetaFields(5).toDouble

          //val stormCategory = stormMetaFields(6)


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
