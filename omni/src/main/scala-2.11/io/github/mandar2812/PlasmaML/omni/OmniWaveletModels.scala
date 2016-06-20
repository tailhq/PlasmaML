package io.github.mandar2812.PlasmaML.omni

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.pipes.{DataPipe, ReversibleScaler, StreamDataPipe}
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.utils.GaussianScaler
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}


/**
  * Created by mandar on 16/6/16.
  */
object OmniWaveletModels {

  var (orderFeat, orderTarget) = (3,3)

  var (trainingStart, trainingEnd) = ("2014/09/15/00", "2014/12/15/00")

  var (validationStart, validationEnd) = ("2008/01/30/00", "2008/06/30/00")

  var (testStart, testEnd) = ("2004/07/21/20", "2004/09/01/00")

  def apply(alpha: Double = 0.01, reg: Double = 0.001,
            momentum: Double = 0.02, maxIt: Int = 20,
            mini: Double = 1.0, useWaveletBasis: Boolean = true) = {


    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val (hFeat, hTarg) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    DateTimeZone.setDefault(DateTimeZone.UTC)
    val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")

    val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H")

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

    val names = Map(
      24 -> "Solar Wind Speed",
      16 -> "I.M.F Bz",
      40 -> "Dst",
      41 -> "AE",
      38 -> "Kp",
      39 -> "Sunspot Number",
      28 -> "Plasma Flow Pressure")

    val column = 40

    val deltaOperationMult = (deltaT: Int, deltaTargets: Int) =>
      DataPipe((lines: Stream[(Double, Double)]) =>
        lines.toList.sliding(deltaT+deltaTargets).map((history) => {
          val features = DenseVector(history.take(deltaT).map(_._2).toArray)
          val outputs = DenseVector(history.takeRight(deltaTargets).map(_._2).toArray)
          (features, outputs)
        }).toStream)

    val preProcessPipe = fileToStream >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        List(0,1,2,column),
        Map(
          16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")
      ) >
      removeMissingLines >
      extractTimeSeries((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/"+hour.toInt.toString)
        dt.getMillis/1000.0
      })

    val trainPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val testPipe = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val postPipe = deltaOperationMult(pF,pT)

    val waveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
      (hFeat*hTarg)(featAndTarg))

    val modelTrainTestWavelet =
      DataPipe((trainTest:
                (Stream[(DenseVector[Double], DenseVector[Double])],
                  Stream[(DenseVector[Double], DenseVector[Double])])) => {


        val gr = FFNeuralGraph(
          trainTest._1.head._1.length,
          trainTest._1.head._2.length,
          0, List("linear"), List())

        val transform = DataPipe(identity[Stream[(DenseVector[Double], DenseVector[Double])]] _)

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], DenseVector[Double])]
          ](trainTest._1, gr, transform)

        model.setLearningRate(alpha)
          .setMaxIterations(maxIt)
          .setRegParam(reg)
          .setMomentum(momentum)
          .setBatchFraction(mini)
          .learn()

        val testSetToResult = DataPipe(
          (testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => model.test(testSet)) >
          StreamDataPipe(
            (tuple: (DenseVector[Double], DenseVector[Double])) => (hTarg.i*hTarg.i)(tuple)) >
          DataPipe((scoresAndLabels: Stream[(DenseVector[Double], DenseVector[Double])]) => {
            val metrics = new MultiRegressionMetrics(
              scoresAndLabels.toList,
              scoresAndLabels.length)
            metrics.setName(names(column)+" "+pT+" hour forecast").print()
          })

        testSetToResult(trainTest._2)
      })

    val modelTrainTest =
      DataPipe((trainTest:
                (Stream[(DenseVector[Double], DenseVector[Double])],
                  Stream[(DenseVector[Double], DenseVector[Double])],
                  (GaussianScaler, GaussianScaler))) => {


        val reverseTargetsScaler = trainTest._3._2.i * trainTest._3._2.i

        val gr = FFNeuralGraph(
          trainTest._1.head._1.length,
          trainTest._1.head._2.length,
          0, List("linear"), List())

        val transform = DataPipe(identity[Stream[(DenseVector[Double], DenseVector[Double])]] _)

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], DenseVector[Double])]
          ](trainTest._1, gr, transform)

        model.setLearningRate(alpha)
          .setMaxIterations(maxIt)
          .setRegParam(reg)
          .setMomentum(momentum)
          .setBatchFraction(mini)
          .learn()

        val testSetToResult = DataPipe(
          (testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => model.test(testSet)) >
          StreamDataPipe(
            (tuple: (DenseVector[Double], DenseVector[Double])) => reverseTargetsScaler(tuple)) >
          DataPipe((scoresAndLabels: Stream[(DenseVector[Double], DenseVector[Double])]) => {
            val metrics = new MultiRegressionMetrics(
              scoresAndLabels.toList,
              scoresAndLabels.length)
            metrics.setName(names(column)+" "+pT+" hour forecast").print()
          })

        testSetToResult(trainTest._2)
      })


    val finalPipe = useWaveletBasis match {
      case true =>
        duplicate(preProcessPipe) >
          DataPipe(trainPipe, testPipe) >
          duplicate(postPipe > waveletPipe) >
          modelTrainTestWavelet

      case false =>
        duplicate(preProcessPipe) >
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
