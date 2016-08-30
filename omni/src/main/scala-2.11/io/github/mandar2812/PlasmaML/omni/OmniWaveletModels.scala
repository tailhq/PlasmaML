package io.github.mandar2812.PlasmaML.omni

import breeze.linalg.{DenseMatrix, DenseVector, pinv}
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.evaluation.{MultiRegressionMetrics, RegressionMetrics}
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.kernels.{CoRegDiracKernel, _}
import io.github.mandar2812.dynaml.models.gp.MOGPRegressionModel
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DataPipe, ReversibleScaler, StreamDataPipe}
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.utils.GaussianScaler
import org.apache.log4j.Logger
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

import scala.collection.mutable.{MutableList => ML}

class CoRegOMNIKernel extends LocalSVMKernel[Int] {
  override val hyper_parameters: List[String] = List()

  override def gradient(x: Int, y: Int): Map[String, Double] = Map()

  override def evaluate(x: Int, y: Int): Double = if(x == y) 1.0 else 0.0
}

class CoRegLaplaceKernel(bandwidth: Double) extends LocalSVMKernel[Int] {

  override val hyper_parameters: List[String] = List("coRegLB")

  state = Map("coRegLB" -> bandwidth)

  override def gradient(x: Int, y: Int): Map[String, Double] =
    Map("coRegLB" -> 1.0*evaluate(x,y)*math.abs(x-y)/math.pow(state("coRegLB"), 2))

  override def evaluate(x: Int, y: Int): Double = {
    math.exp(-1.0*math.abs(x-y)/state("coRegLB"))
  }
}


/**
  * Created by mandar on 16/6/16.
  */
object OmniWaveletModels {

  DateTimeZone.setDefault(DateTimeZone.UTC)
  val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")

  var (orderFeat, orderTarget) = (4,2)

  var (trainingStart, trainingEnd) = ("2014/02/15/00", "2014/02/25/23")

  var (validationStart, validationEnd) = ("2008/01/30/00", "2008/06/30/00")

  var (testStart, testEnd) = ("2004/07/21/20", "2004/08/02/00")

  var hidden_layers:Int = 1

  var neuronCounts: List[Int] = List(6)

  var neuronActivations: List[String] = List("logsig", "linear")

  var column: Int = 40

  var exogenousInputs: List[Int] = List()

  var globalOpt: String = "GS"

  val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H")

  def preProcess = fileToStream >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      List(0,1,2,column)++exogenousInputs,
      Map(
        16 -> "999.9", 21 -> "999.9",
        24 -> "9999.", 23 -> "999.9",
        40 -> "99999", 22 -> "9999999.",
        25 -> "999.9", 28 -> "99.99",
        27 -> "9.999", 39 -> "999",
        45 -> "99999.99", 46 -> "99999.99",
        47 -> "99999.99")
    ) >
    removeMissingLines

  val deltaOperationMult = (deltaT: Int, deltaTargets: Int) =>
    DataPipe((lines: Stream[(Double, Double)]) =>
      lines.toList.sliding(deltaT+deltaTargets+1).map((history) => {
        val features = DenseVector(history.take(deltaT).map(_._2).toArray)
        val outputs = DenseVector(history.takeRight(deltaTargets).map(_._2).toArray)
        (features, outputs)
      }).toStream)

  val deltaOperationARXMult = (deltaT: List[Int], deltaTargets: Int) =>
    DataPipe((lines: Stream[(Double, DenseVector[Double])]) =>
      lines.toList.sliding(deltaT.max+deltaTargets+1).map((history) => {

        val hist = history.take(deltaT.max).map(_._2)
        val histOut = history.takeRight(deltaTargets).map(_._2)

        val featuresAcc: ML[Double] = ML()

        (0 until hist.head.length).foreach((dimension) => {
          //for each dimension/regressor take points t to t-order
          featuresAcc ++= hist.takeRight(deltaT(dimension))
            .map(vec => vec(dimension))
        })

        val outputs = DenseVector(histOut.map(_(0)).toArray)
        val features = DenseVector(featuresAcc.toArray)

        (features, outputs)
      }).toStream)

  val names = Map(
    24 -> "Solar Wind Speed", 16 -> "I.M.F Bz",
    40 -> "Dst", 41 -> "AE",
    38 -> "Kp", 39 -> "Sunspot Number",
    28 -> "Plasma Flow Pressure")

  var useWaveletBasis: Boolean = true

  var deltaT: List[Int] = List()

  def train(kernel: LocalScalarKernel[(DenseVector[Double], Int)],
            noise: LocalScalarKernel[(DenseVector[Double], Int)],
            grid: Int, step: Double, useLogSc: Boolean, maxIt:Int):
  (MOGPRegressionModel[DenseVector[Double]], (GaussianScaler, GaussianScaler)) = {

    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val arxOrders = if(deltaT.isEmpty) List.fill[Int](exogenousInputs.length+1)(pF) else deltaT

    val (hFeat, _) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val haarWaveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
      if (useWaveletBasis)
        (DenseVector(featAndTarg._1
          .toArray
          .grouped(pF)
          .map(l => hFeat(DenseVector(l)).toArray)
          .reduceLeft((a,b) => a ++ b)),
          featAndTarg._2)
      else
        featAndTarg
    )

    val (trainingStartDate, trainingEndDate) =
        (formatter.parseDateTime(trainingStart).minusHours(pF),
        formatter.parseDateTime(trainingEnd).plusHours(pT))

    val (trStampStart, trStampEnd) =
      (trainingStartDate.getMillis/1000.0, trainingEndDate.getMillis/1000.0)

    val filterData = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val filterDataARX = StreamDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val prepareFeaturesAndOutputs = if(exogenousInputs.isEmpty) {
      extractTimeSeries((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString)
        dt.getMillis/1000.0 }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString)
        dt.getMillis/1000.0 }) >
      filterDataARX >
      deltaOperationARXMult(arxOrders, pT)
    }

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      gaussianScaling >
      DataPipe((dataAndScales: (
        Stream[(DenseVector[Double], DenseVector[Double])],
          (GaussianScaler, GaussianScaler))) => {

        val model = new MOGPRegressionModel[DenseVector[Double]](
          kernel, noise, dataAndScales._1,
          dataAndScales._1.length, pT)

        val gs = globalOpt match {
          case "CSA" =>
            new CoupledSimulatedAnnealing[model.type](model)
              .setGridSize(grid)
              .setStepSize(step)
              .setLogScale(useLogSc)
              .setMaxIterations(maxIt)
              .setVariant(CoupledSimulatedAnnealing.MwVC)
          case "GS" =>
            new GridSearch[model.type](model)
              .setGridSize(grid)
              .setStepSize(step)
              .setLogScale(useLogSc)
        }

        val startConf = kernel.effective_state ++ noise.effective_state

        val (tunedGP, _) = gs.optimize(startConf)

        (tunedGP, dataAndScales._2)
      })


    flow("data/omni2_"+trainingStartDate.getYear+".csv")

  }

  def test(model: MOGPRegressionModel[DenseVector[Double]], sc: (GaussianScaler, GaussianScaler)) = {

    val (pF, pT) = (math.pow(2,orderFeat).toInt,math.pow(2, orderTarget).toInt)
    val arxOrders = if(deltaT.isEmpty) List.fill[Int](exogenousInputs.length+1)(pF) else deltaT

    val (hFeat, _) = (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val haarWaveletPipe = StreamDataPipe((featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
      if (useWaveletBasis)
        (DenseVector(featAndTarg._1
          .toArray
          .grouped(pF)
          .map(l => hFeat(DenseVector(l)).toArray)
          .reduceLeft((a,b) => a ++ b)),
          featAndTarg._2)
      else
        featAndTarg
    )

    val (testStartDate, testEndDate) =
      (formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT))

    val (tStampStart, tStampEnd) = (testStartDate.getMillis/1000.0, testEndDate.getMillis/1000.0)

    val filterData = StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val filterDataARX = StreamDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= tStampStart && couple._1 <= tStampEnd)

    val prepareFeaturesAndOutputs = if(exogenousInputs.isEmpty) {
      extractTimeSeries((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString)
        dt.getMillis/1000.0 }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year,day,hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString)
        dt.getMillis/1000.0 }) >
        filterDataARX >
        deltaOperationARXMult(arxOrders, pT)
    }

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      DataPipe((testDat: Stream[(DenseVector[Double], DenseVector[Double])]) => (sc._1*sc._2)(testDat)) >
      DataPipe((nTestDat: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        model.test(nTestDat)
          .map(t => (t._1._2, (t._3, t._2)))
          .groupBy(_._1).toSeq
          .sortBy(_._1)
          .map(res =>
            new RegressionMetrics(
              res._2.map(_._2).toList.map(c => {
                val rescaler = (sc._2.mean(res._1), sc._2.sigma(res._1))
                ((c._1 * rescaler._2) + rescaler._1, (c._2 * rescaler._2) + rescaler._1)
              }),
              res._2.length).setName("Dst "+(res._1+1).toString+" hours ahead")
          )
      })

    flow("data/omni2_"+testStartDate.getYear+".csv")
  }

  def train(alpha: Double = 0.01, reg: Double = 0.001,
            momentum: Double = 0.02, maxIt: Int = 20,
            mini: Double = 1.0, useWaveletBasis: Boolean = true)
  : (FeedForwardNetwork[Stream[(DenseVector[Double], DenseVector[Double])]],
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

    val modelTrain =
      DataPipe((trainTest:
                (Stream[(DenseVector[Double], DenseVector[Double])],
                  (GaussianScaler, GaussianScaler))) => {

        val gr = FFNeuralGraph(
          trainTest._1.head._1.length,
          trainTest._1.head._2.length,
          hidden_layers, neuronActivations,
          neuronCounts)

        implicit val transform = DataPipe(identity[Stream[(DenseVector[Double], DenseVector[Double])]] _)

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], DenseVector[Double])]
          ](trainTest._1, gr)

        model.setLearningRate(alpha)
          .setMaxIterations(maxIt)
          .setRegParam(reg)
          .setMomentum(momentum)
          .setBatchFraction(mini)
          .learn()

        (model, trainTest._2)
      })

    val finalPipe = prepareTrainingData >
      gaussianScaling >
      modelTrain

    finalPipe(20)

  }

  def test(model: FeedForwardNetwork[Stream[(DenseVector[Double], DenseVector[Double])]],
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
        (testSet: Stream[(DenseVector[Double], DenseVector[Double])]) => model.test(testSet)) >
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

    OmniWaveletModels.exogenousInputs = List()

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

        val gr = FFNeuralGraph(
          trainTest._1.head._1.length,
          trainTest._1.head._2.length,
          hidden_layers, neuronActivations,
          neuronCounts)

        implicit val transform = DataPipe(identity[Stream[(DenseVector[Double], DenseVector[Double])]] _)

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], DenseVector[Double])]
          ](trainTest._1, gr)

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

        val gr = FFNeuralGraph(
          trainTest._1.head._1.length,
          trainTest._1.head._2.length,
          1, List("logsig","linear"),
          List(4))

        implicit val transform = DataPipe(identity[Stream[(DenseVector[Double], DenseVector[Double])]] _)

        val model = new FeedForwardNetwork[
          Stream[(DenseVector[Double], DenseVector[Double])]
          ](trainTest._1, gr)

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

object DstPersistenceMOExperiment {

  val logger = Logger.getLogger(this.getClass)

  def apply(orderT: Int) = {
    OmniWaveletModels.orderTarget = orderT

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

          //val minDst = stormMetaFields(5).toDouble

          //val stormCategory = stormMetaFields(6)


          OmniWaveletModels.testStart = startDate+"/"+startHour
          OmniWaveletModels.testEnd = endDate+"/"+endHour

          logger.info("Testing on Storm: "+OmniWaveletModels.testStart+" to "+OmniWaveletModels.testEnd)

          OmniWaveletModels.test()
        }) >
        DataPipe((metrics: Stream[MultiRegressionMetrics]) =>
          metrics.reduce((m,n) => m++n))

    stormsPipe("data/geomagnetic_storms.csv")

  }
}


object DstWaveletExperiment {

  val logger = Logger.getLogger(this.getClass)

  var learningRate: Double = 1.0

  var reg: Double = 0.0005

  var momentum: Double = 0.6

  var it:Int = 150

  def apply(orderF: Int = 4, orderT: Int = 3, useWavelets: Boolean = true) = {

    OmniWaveletModels.orderFeat = orderF
    OmniWaveletModels.orderTarget = orderT
    OmniWaveletModels.useWaveletBasis = useWavelets

    val (model, scaler) = OmniWaveletModels.train(learningRate, reg, momentum, it, 1.0)

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


          OmniWaveletModels.testStart = startDate+"/"+startHour
          OmniWaveletModels.testEnd = endDate+"/"+endHour

          logger.info("Testing on Storm: "+OmniWaveletModels.testStart+" to "+OmniWaveletModels.testEnd)

          OmniWaveletModels.test(model, scaler)
        }) >
        DataPipe((metrics: Stream[MultiRegressionMetrics]) =>
          metrics.reduce((m,n) => m++n))

    stormsPipe("data/geomagnetic_storms.csv")

  }
}

object DstMOGPExperiment {

  val logger = Logger.getLogger(this.getClass)

  var gridSize = 3
  var gridStep = 0.2
  var logScale = false
  var maxIt = 40


  def apply(orderF: Int = 4, orderT: Int = 3, useWavelets: Boolean = true)(
    kernel: CompositeCovariance[(DenseVector[Double], Int)],
    noise: CompositeCovariance[(DenseVector[Double], Int)]) = {

    OmniWaveletModels.orderFeat = orderF
    OmniWaveletModels.orderTarget = orderT
    OmniWaveletModels.useWaveletBasis = useWavelets



    OmniWaveletModels.orderFeat = orderF
    OmniWaveletModels.orderTarget = orderT

    val (model, scaler) = OmniWaveletModels.train(kernel, noise, gridSize, gridStep, useLogSc = logScale, maxIt)

    model.persist()

    //val (model, scaler) = OmniWaveletModels.train(learningRate, reg, momentum, it, 1.0)

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        //DataPipe((st: Stream[String]) => st.take(43)) >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val eventId = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          //val minDst = stormMetaFields(5).toDouble

          //val stormCategory = stormMetaFields(6)

          OmniWaveletModels.testStart = startDate+"/"+startHour
          OmniWaveletModels.testEnd = endDate+"/"+endHour

          logger.info("Testing on Storm: "+OmniWaveletModels.testStart+" to "+OmniWaveletModels.testEnd)

          OmniWaveletModels.test(model, scaler)
        }) >
        DataPipe((metrics: Stream[Iterable[RegressionMetrics]]) =>
          metrics.reduceLeft((m,n) => m.zip(n).map(pair => pair._1 ++ pair._2)))

    stormsPipe("data/geomagnetic_storms.csv")

  }
}
