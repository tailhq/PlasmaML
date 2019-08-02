package io.github.mandar2812.PlasmaML.omni

import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.evaluation.{
  BinaryClassificationMetrics,
  MultiRegressionMetrics,
  RegressionMetrics
}
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.ContinuousProcessModel
import io.github.mandar2812.dynaml.models.gp.{
  AbstractGPRegressionModel,
  KroneckerMOGPModel,
  MOGPRegressionModel
}
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.models.stp.MOStudentTRegression
import io.github.mandar2812.dynaml.optimization._
import io.github.mandar2812.dynaml.pipes.{DataPipe, IterableDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.wavelets.{
  GroupedHaarWaveletFilter,
  HaarWaveletFilter
}
import org.apache.log4j.Logger
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

import scala.collection.mutable.{MutableList => ML}

/**
  * Created by mandar on 16/6/16.
  */
object OmniMultiOutputModels {

  DateTimeZone.setDefault(DateTimeZone.UTC)
  val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")

  var (orderFeat, orderTarget) = (4, 2)

  var (trainingStart, trainingEnd) = ("2014/01/10/00", "2014/12/30/23")

  var (validationStart, validationEnd) = ("2008/01/30/00", "2008/06/30/00")

  var (testStart, testEnd) = ("2004/07/21/20", "2004/08/02/00")

  var hidden_layers: Int = 2

  var neuronCounts: List[Int] = List(8, 6)

  var neuronActivations: List[String] = List("logsig", "linear")

  var activations = List(MagicSELU, MagicSELU, VectorLinear)

  var column: Int = 40

  var exogenousInputs: List[Int] = List()

  var globalOpt: String = "GS"

  val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H")

  def preProcess =
    fileToStream >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        List(0, 1, 2, column) ++ exogenousInputs,
        Map(
          16 -> "999.9",
          21 -> "999.9",
          24 -> "9999.",
          23 -> "999.9",
          40 -> "99999",
          22 -> "9999999.",
          25 -> "999.9",
          28 -> "99.99",
          27 -> "9.999",
          39 -> "999",
          45 -> "99999.99",
          46 -> "99999.99",
          47 -> "99999.99"
        )
      ) >
      removeMissingLines

  val deltaOperationMult = (deltaT: Int, deltaTargets: Int) =>
    DataPipe(
      (lines: Iterable[(Double, Double)]) =>
        lines.toList
          .sliding(deltaT + deltaTargets)
          .map((history) => {
            val features = DenseVector(history.take(deltaT).map(_._2).toArray)
            val outputs =
              DenseVector(history.takeRight(deltaTargets).map(_._2).toArray)
            (features, outputs)
          })
          .toStream
    )

  val deltaOperationARXMult = (deltaT: List[Int], deltaTargets: Int) =>
    DataPipe(
      (lines: Iterable[(Double, DenseVector[Double])]) =>
        lines.toList
          .sliding(deltaT.max + deltaTargets)
          .map((history) => {

            val hist    = history.take(deltaT.max).map(_._2)
            val histOut = history.takeRight(deltaTargets).map(_._2)

            val featuresAcc: ML[Double] = ML()

            (0 until hist.head.length).foreach((dimension) => {
              //for each dimension/regressor take points t to t-order
              featuresAcc ++= hist
                .takeRight(deltaT(dimension))
                .map(vec => vec(dimension))
            })

            val outputs  = DenseVector(histOut.map(_(0)).toArray)
            val features = DenseVector(featuresAcc.toArray)

            (features, outputs)
          })
          .toStream
    )

  val names = Map(
    24 -> "Solar Wind Speed",
    16 -> "I.M.F Bz",
    40 -> "Dst",
    41 -> "AE",
    38 -> "Kp",
    39 -> "Sunspot Number",
    28 -> "Plasma Flow Pressure"
  )

  var useWaveletBasis: Boolean = true

  var deltaT: List[Int] = List()

  var threshold = -70.0

  var numStorms = 6

  val numStormsStart = 4

  def gHFeat =
    GroupedHaarWaveletFilter(Array.fill(exogenousInputs.length + 1)(orderFeat))
  def gHTarg = HaarWaveletFilter(orderTarget)

  def haarWaveletPipe =
    IterableDataPipe(
      (featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
        if (useWaveletBasis)
          (gHFeat(featAndTarg._1), featAndTarg._2)
        else
          featAndTarg
    )

  def trainStorms(
    kernel: LocalScalarKernel[(DenseVector[Double], Int)],
    noise: LocalScalarKernel[(DenseVector[Double], Int)],
    grid: Int,
    step: Double,
    useLogSc: Boolean,
    maxIt: Int
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val prepareTrainingData = (n: Int) => {

      val stormsPipe =
        fileToStream >
          replaceWhiteSpaces >
          IterableDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId   = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            (startDate + "/" + startHour, endDate + "/" + endHour)
          }) >
          DataPipe(
            (s: Iterable[(String, String)]) =>
              s.take(numStormsStart) ++ s.takeRight(n) ++
                /*Stream(("2015/03/17/00", "2015/03/18/23")) ++*/
                Stream(("2015/06/22/08", "2015/06/23/20")) ++
                Stream(("2008/01/02/00", "2008/01/03/00"))
          ) >
          IterableDataPipe((storm: (String, String)) => {
            // for each storm construct a data set

            val (trainingStartDate, trainingEndDate) =
              (
                formatter.parseDateTime(storm._1).minusHours(pF),
                formatter.parseDateTime(storm._2).plusHours(pT)
              )

            val (trStampStart, trStampEnd) =
              (
                trainingStartDate.getMillis / 1000.0,
                trainingEndDate.getMillis / 1000.0
              )

            val filterData = IterableDataPipe(
              (couple: (Double, Double)) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val filterDataARX = IterableDataPipe(
              (couple: (Double, DenseVector[Double])) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
              extractTimeSeries((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterData >
                deltaOperationMult(arxOrders.head, pT)
            } else {
              extractTimeSeriesVec((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterDataARX >
                deltaOperationARXMult(arxOrders, pT)
            }

            val getTraining = preProcess >
              prepareFeaturesAndOutputs >
              haarWaveletPipe

            getTraining("data/omni2_" + trainingStartDate.getYear + ".csv")

          }) >
          DataPipe(
            (s: Iterable[
              Iterable[(DenseVector[Double], DenseVector[Double])]
            ]) => {
              s.reduce((p, q) => p ++ q)
            }
          )

      stormsPipe("data/geomagnetic_storms2.csv")
    }

    val modelTuning = (dataAndScales: (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )) => {

      val training_data = dataAndScales._1.toStream

      val model = new MOGPRegressionModel[DenseVector[Double]](
        kernel,
        noise,
        training_data,
        training_data.length,
        pT
      )

      val gs = globalOpt match {
        case "CSA" =>
          new CoupledSimulatedAnnealing[AbstractGPRegressionModel[
            Stream[(DenseVector[Double], DenseVector[Double])],
            (DenseVector[Double], Int)
          ]](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
        case "GS" =>
          new GridSearch[AbstractGPRegressionModel[
            Stream[(DenseVector[Double], DenseVector[Double])],
            (DenseVector[Double], Int)
          ]](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)

        case "GPC" =>
          new ProbGPCommMachine[Stream[
            (DenseVector[Double], DenseVector[Double])
          ], (DenseVector[Double], Int)](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)

        case "ML-II" =>
          new GradBasedGlobalOptimizer[AbstractGPRegressionModel[Stream[
            (DenseVector[Double], DenseVector[Double])
          ], (DenseVector[Double], Int)]](model)
      }

      val startConf = kernel.effective_state ++ noise.effective_state

      val (tunedGP, _) = gs.optimize(startConf)

      (tunedGP, dataAndScales._2)
    }

    (DataPipe(prepareTrainingData) >
      gaussianScaling >
      DataPipe(modelTuning))(numStorms)

  }

  def trainStormsKron(
    kernel: LocalScalarKernel[DenseVector[Double]],
    noise: LocalScalarKernel[DenseVector[Double]],
    coRegK: LocalScalarKernel[Int],
    grid: Int,
    step: Double,
    useLogSc: Boolean,
    maxIt: Int
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val prepareTrainingData = (n: Int) => {

      val stormsPipe =
        fileToStream >
          replaceWhiteSpaces >
          IterableDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId   = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            (startDate + "/" + startHour, endDate + "/" + endHour)
          }) >
          DataPipe(
            (s: Iterable[(String, String)]) =>
              s.take(numStormsStart) ++ s.takeRight(n) ++
                /*Stream(("2015/03/17/00", "2015/03/18/23")) ++*/
                Stream(("2015/06/22/08", "2015/06/23/20")) ++
                Stream(("2008/01/02/00", "2008/01/03/00"))
          ) >
          IterableDataPipe((storm: (String, String)) => {
            // for each storm construct a data set

            val (trainingStartDate, trainingEndDate) =
              (
                formatter.parseDateTime(storm._1).minusHours(pF),
                formatter.parseDateTime(storm._2).plusHours(pT)
              )

            val (trStampStart, trStampEnd) =
              (
                trainingStartDate.getMillis / 1000.0,
                trainingEndDate.getMillis / 1000.0
              )

            val filterData = IterableDataPipe(
              (couple: (Double, Double)) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val filterDataARX = IterableDataPipe(
              (couple: (Double, DenseVector[Double])) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
              extractTimeSeries((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterData >
                deltaOperationMult(arxOrders.head, pT)
            } else {
              extractTimeSeriesVec((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterDataARX >
                deltaOperationARXMult(arxOrders, pT)
            }

            val getTraining = preProcess >
              prepareFeaturesAndOutputs >
              haarWaveletPipe

            getTraining("data/omni2_" + trainingStartDate.getYear + ".csv")

          }) >
          DataPipe(
            (s: Iterable[
              Iterable[(DenseVector[Double], DenseVector[Double])]
            ]) => {
              s.reduce((p, q) => p ++ q)
            }
          )

      stormsPipe("data/geomagnetic_storms2.csv")
    }

    val modelTuning = (dataAndScales: (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )) => {

      val training_data = dataAndScales._1.toStream
      val model = new KroneckerMOGPModel[DenseVector[Double]](
        kernel,
        noise,
        coRegK,
        training_data,
        training_data.length,
        pT
      )

      val gs = globalOpt match {
        case "CSA" =>
          new CoupledSimulatedAnnealing[AbstractGPRegressionModel[
            Stream[(DenseVector[Double], DenseVector[Double])],
            (DenseVector[Double], Int)
          ]](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
        case "GS" =>
          new GridSearch[AbstractGPRegressionModel[
            Stream[(DenseVector[Double], DenseVector[Double])],
            (DenseVector[Double], Int)
          ]](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)

        case "GPC" =>
          new ProbGPCommMachine[Stream[
            (DenseVector[Double], DenseVector[Double])
          ], (DenseVector[Double], Int)](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)

        case "ML-II" =>
          new GradBasedGlobalOptimizer[AbstractGPRegressionModel[Stream[
            (DenseVector[Double], DenseVector[Double])
          ], (DenseVector[Double], Int)]](model)
      }

      val startConf = model.covariance.effective_state ++ model.noiseModel.effective_state

      val (tunedGP, _) = gs.optimize(startConf)

      (tunedGP, dataAndScales._2)
    }

    (DataPipe(prepareTrainingData) >
      gaussianScaling >
      DataPipe(modelTuning))(numStorms)

  }

  def trainSTPStorms(
    kernel: LocalScalarKernel[(DenseVector[Double], Int)],
    noise: LocalScalarKernel[(DenseVector[Double], Int)],
    mu: Double,
    grid: Int,
    step: Double,
    useLogSc: Boolean,
    maxIt: Int
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val prepareTrainingData = (n: Int) => {

      val stormsPipe =
        fileToStream >
          replaceWhiteSpaces >
          IterableDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId   = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            (startDate + "/" + startHour, endDate + "/" + endHour)
          }) >
          DataPipe(
            (s: Iterable[(String, String)]) =>
              s.take(numStormsStart) ++ s.takeRight(n) ++
                Stream(("2015/03/17/00", "2015/03/18/23")) ++
                Stream(("2015/06/22/08", "2015/06/23/20")) ++
                Stream(("2008/01/02/00", "2008/02/02/00"))
          ) >
          IterableDataPipe((storm: (String, String)) => {
            // for each storm construct a data set

            val (trainingStartDate, trainingEndDate) =
              (
                formatter.parseDateTime(storm._1).minusHours(pF),
                formatter.parseDateTime(storm._2).plusHours(pT)
              )

            val (trStampStart, trStampEnd) =
              (
                trainingStartDate.getMillis / 1000.0,
                trainingEndDate.getMillis / 1000.0
              )

            val filterData = IterableDataPipe(
              (couple: (Double, Double)) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val filterDataARX = IterableDataPipe(
              (couple: (Double, DenseVector[Double])) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
              extractTimeSeries((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterData >
                deltaOperationMult(arxOrders.head, pT)
            } else {
              extractTimeSeriesVec((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterDataARX >
                deltaOperationARXMult(arxOrders, pT)
            }

            val getTraining = preProcess >
              prepareFeaturesAndOutputs >
              haarWaveletPipe

            getTraining("data/omni2_" + trainingStartDate.getYear + ".csv")

          }) >
          DataPipe(
            (s: Iterable[
              Iterable[(DenseVector[Double], DenseVector[Double])]
            ]) => {
              s.reduce((p, q) => p ++ q)
            }
          )

      stormsPipe("data/geomagnetic_storms2.csv")
    }

    val modelTuning = (dataAndScales: (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )) => {

      val training_data = dataAndScales._1.toStream
      val model = new MOStudentTRegression[DenseVector[Double]](
        mu,
        kernel,
        noise,
        training_data,
        training_data.length,
        pT
      )

      val gs = globalOpt match {
        case "CSA" =>
          new CoupledSimulatedAnnealing[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
        case "GS" =>
          new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
      }

      val startConf = kernel.effective_state ++ noise.effective_state ++ Map(
        "degrees_of_freedom" -> mu
      )

      val (tunedGP, _) = gs.optimize(startConf)

      (tunedGP, dataAndScales._2)
    }

    (DataPipe(prepareTrainingData) >
      gaussianScaling >
      DataPipe(modelTuning))(numStorms)

  }

  def trainStormsGrad(
    kernel: LocalScalarKernel[(DenseVector[Double], Int)],
    noise: LocalScalarKernel[(DenseVector[Double], Int)],
    grid: Int,
    step: Double,
    useLogSc: Boolean,
    maxIt: Int
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val prepareTrainingData = (n: Int) => {

      val stormsPipe =
        fileToStream >
          replaceWhiteSpaces >
          IterableDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId   = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            (startDate + "/" + startHour, endDate + "/" + endHour)
          }) >
          DataPipe(
            (s: Iterable[(String, String)]) =>
              s.take(numStormsStart) ++ s.takeRight(n) ++
                Stream(("2015/03/17/00", "2015/03/18/23")) ++
                Stream(("2015/06/22/08", "2015/06/23/20")) ++
                Stream(("2008/01/02/00", "2008/02/02/00"))
          ) >
          IterableDataPipe((storm: (String, String)) => {
            // for each storm construct a data set

            val (trainingStartDate, trainingEndDate) =
              (
                formatter.parseDateTime(storm._1).minusHours(pF),
                formatter.parseDateTime(storm._2).plusHours(pT)
              )

            val (trStampStart, trStampEnd) =
              (
                trainingStartDate.getMillis / 1000.0,
                trainingEndDate.getMillis / 1000.0
              )

            val filterData = IterableDataPipe(
              (couple: (Double, Double)) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val filterDataARX = IterableDataPipe(
              (couple: (Double, DenseVector[Double])) =>
                couple._1 >= trStampStart && couple._1 <= trStampEnd
            )

            val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
              extractTimeSeries((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterData >
                deltaOperationMult(arxOrders.head, pT)
            } else {
              extractTimeSeriesVec((year, day, hour) => {
                val dt = dayofYearformat.parseDateTime(
                  year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
                )
                dt.getMillis / 1000.0
              }) >
                filterDataARX >
                deltaOperationARXMult(arxOrders, pT)
            }

            val getTraining = preProcess >
              prepareFeaturesAndOutputs >
              haarWaveletPipe

            getTraining("data/omni2_" + trainingStartDate.getYear + ".csv")

          }) >
          DataPipe(
            (s: Iterable[
              Iterable[(DenseVector[Double], DenseVector[Double])]
            ]) => {
              s.reduce((p, q) => p ++ q)
            }
          )

      stormsPipe("data/geomagnetic_storms2.csv")
    }

    val modelTuning = (dataAndScales: (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )) => {

      val training_data = dataAndScales._1.toStream
      val model = new MOGPRegressionModel[DenseVector[Double]](
        kernel,
        noise,
        training_data,
        training_data.length,
        pT
      )

      val gs = globalOpt match {
        case "CSA" =>
          new CoupledSimulatedAnnealing[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
        case "GS" =>
          new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
      }

      val startConf = kernel.effective_state ++ noise.effective_state

      val (tunedGP, _) = gs.optimize(startConf)

      (tunedGP, dataAndScales._2)
    }

    (DataPipe(prepareTrainingData) >
      gaussianScaling >
      DataPipe(modelTuning))(numStorms)

  }

  def train(
    kernel: LocalScalarKernel[(DenseVector[Double], Int)],
    noise: LocalScalarKernel[(DenseVector[Double], Int)],
    grid: Int,
    step: Double,
    useLogSc: Boolean,
    maxIt: Int
  ): (
    MOGPRegressionModel[DenseVector[Double]],
    (GaussianScaler, GaussianScaler)
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val (trainingStartDate, trainingEndDate) =
      (
        formatter.parseDateTime(trainingStart).minusHours(pF),
        formatter.parseDateTime(trainingEnd).plusHours(pT)
      )

    val (trStampStart, trStampEnd) =
      (trainingStartDate.getMillis / 1000.0, trainingEndDate.getMillis / 1000.0)

    val filterData = IterableDataPipe(
      (couple: (Double, Double)) =>
        couple._1 >= trStampStart && couple._1 <= trStampEnd
    )

    val filterDataARX = IterableDataPipe(
      (couple: (Double, DenseVector[Double])) =>
        couple._1 >= trStampStart && couple._1 <= trStampEnd
    )

    val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
      extractTimeSeries((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterDataARX >
        deltaOperationARXMult(arxOrders, pT)
    }

    val modelTuning = (dataAndScales: (
      Iterable[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler)
    )) => {

      val training_data = dataAndScales._1.toStream
      val model = new MOGPRegressionModel[DenseVector[Double]](
        kernel,
        noise,
        training_data,
        training_data.length,
        pT
      )

      val gs = globalOpt match {
        case "CSA" =>
          new CoupledSimulatedAnnealing[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
            .setMaxIterations(maxIt)
            .setVariant(AbstractCSA.MwVC)
        case "GS" =>
          new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(useLogSc)
      }

      val startConf = kernel.effective_state ++ noise.effective_state

      val (tunedGP, _) = gs.optimize(startConf)

      (tunedGP, dataAndScales._2)
    }

    (preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      gaussianScaling >
      DataPipe(modelTuning)) run ("data/omni2_" + trainingStartDate.getYear + ".csv")

  }

  def test[
    M <: ContinuousProcessModel[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int), Double, _]
  ](model: M,
    sc: (GaussianScaler, GaussianScaler)
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val haarWaveletPipe = IterableDataPipe(
      (featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
        if (useWaveletBasis)
          (
            DenseVector(
              featAndTarg._1.toArray
                .grouped(pF)
                .map(l => hFeat(DenseVector(l)).toArray)
                .reduceLeft((a, b) => a ++ b)
            ),
            featAndTarg._2
          )
        else
          featAndTarg
    )

    val (testStartDate, testEndDate) =
      (
        formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT)
      )

    val (tStampStart, tStampEnd) =
      (testStartDate.getMillis / 1000.0, testEndDate.getMillis / 1000.0)

    val filterData = IterableDataPipe(
      (couple: (Double, Double)) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val filterDataARX = IterableDataPipe(
      (couple: (Double, DenseVector[Double])) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
      extractTimeSeries((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterDataARX >
        deltaOperationARXMult(arxOrders, pT)
    }

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      DataPipe(
        (testDat: Iterable[(DenseVector[Double], DenseVector[Double])]) =>
          (sc._1 * sc._2)(testDat)
      ) >
      DataPipe(
        (nTestDat: Iterable[(DenseVector[Double], DenseVector[Double])]) => {
          model
            .test(nTestDat)
            .map(t => (t._1._2, (t._3, t._2)))
            .groupBy(_._1)
            .toSeq
            .sortBy(_._1)
            .map(
              res =>
                new RegressionMetrics(
                  res._2
                    .map(_._2)
                    .toList
                    .map(c => {
                      val rescaler = (sc._2.mean(res._1), sc._2.sigma(res._1))
                      (
                        (c._1 * rescaler._2) + rescaler._1,
                        (c._2 * rescaler._2) + rescaler._1
                      )
                    }),
                  res._2.length
                ).setName("Dst " + (res._1 + 1).toString + " hours ahead")
            )
        }
      )

    flow("data/omni2_" + testStartDate.getYear + ".csv")
  }

  def testOnset(
    model: AbstractGPRegressionModel[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int)],
    sc: (GaussianScaler, GaussianScaler)
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val haarWaveletPipe = IterableDataPipe(
      (featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
        if (useWaveletBasis)
          (
            DenseVector(
              featAndTarg._1.toArray
                .grouped(pF)
                .map(l => hFeat(DenseVector(l)).toArray)
                .reduceLeft((a, b) => a ++ b)
            ),
            featAndTarg._2
          )
        else
          featAndTarg
    )

    val (testStartDate, testEndDate) =
      (
        formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT)
      )

    val (tStampStart, tStampEnd) =
      (testStartDate.getMillis / 1000.0, testEndDate.getMillis / 1000.0)

    val filterData = IterableDataPipe(
      (couple: (Double, Double)) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val filterDataARX = IterableDataPipe(
      (couple: (Double, DenseVector[Double])) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
      extractTimeSeries((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterDataARX >
        deltaOperationARXMult(arxOrders, pT)
    }

    val postProcessPipe =
      DataPipe(
        (nTestDat: Iterable[(DenseVector[Double], DenseVector[Double])]) => {
          model
            .test(nTestDat)
            .map(t => (t._1, (t._3, t._2, t._4)))
            .groupBy(_._1._2)
            .toSeq
            .sortBy(_._1)
            .map(res => {
              logger.info("Collating results for Hour t + " + (res._1 + 1))
              val targetIndex = res._1
              val scAndLabel = res._2.map(pattern => {
                val unprocessed_features = pattern._1._1

                val dst_t = if (useWaveletBasis) {
                  val trancatedSc = GaussianScaler(
                    sc._1.mean(0 until pF),
                    sc._1.sigma(0 until pF)
                  )

                  val features_processed =
                    (trancatedSc.i > hFeat.i)(unprocessed_features(0 until pF))
                  features_processed(pF - 1)
                } else {
                  val features_processed = sc._1.i(unprocessed_features)
                  features_processed(pF - 1)
                }

                val (resMean, resSigma) =
                  (sc._2.mean(targetIndex), sc._2.sigma(targetIndex))

                val (predictedMean, actualval, sigma) =
                  (
                    resSigma * pattern._2._1 + resMean,
                    resSigma * pattern._2._2 + resMean,
                    resSigma * (pattern._2._1 - pattern._2._3)
                  )

                val label = if ((actualval - dst_t) <= threshold) 1.0 else 0.0

                val normalDist = Gaussian(predictedMean - dst_t, sigma)
                (normalDist.cdf(threshold), label)

              })
              new BinaryClassificationMetrics(
                scAndLabel.toList,
                scAndLabel.length,
                true
              )
            })
        }
      )

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      DataPipe(
        (testDat: Iterable[(DenseVector[Double], DenseVector[Double])]) =>
          (sc._1 * sc._2)(testDat)
      ) >
      postProcessPipe

    flow("data/omni2_" + testStartDate.getYear + ".csv")
  }

  def generateOnsetPredictions(
    model: AbstractGPRegressionModel[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int)],
    sc: (GaussianScaler, GaussianScaler),
    predictionIndex: Int = 3
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    //val (hFeat, invHFeat) = (haarWaveletFilter(orderFeat), invHaarWaveletFilter(orderFeat))

    val (testStartDate, testEndDate) =
      (
        formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT)
      )

    val (tStampStart, tStampEnd) =
      (testStartDate.getMillis / 1000.0, testEndDate.getMillis / 1000.0)

    val filterData = IterableDataPipe(
      (couple: (Double, Double)) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val filterDataARX = IterableDataPipe(
      (couple: (Double, DenseVector[Double])) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
      extractTimeSeries((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterDataARX >
        deltaOperationARXMult(arxOrders, pT)
    }

    val postProcessPipe =
      DataPipe(
        (nTestDat: Iterable[(DenseVector[Double], DenseVector[Double])]) => {
          model
            .test(nTestDat)
            .filter(_._1._2 == predictionIndex)
            .map(t => (t._1, (t._3, t._2, t._4)))
            .map(res => {
              val pattern = res

              val dst_t = if (useWaveletBasis) {
                val features_processed = (sc._1.i > gHFeat.i)(pattern._1._1)
                //println("Features: "+features_processed)
                features_processed(pF - 1)
              } else {
                val features_processed = sc._1.i(pattern._1._1)
                //println("Features: "+features_processed)
                features_processed(arxOrders.head - 1)
              }

              val (resMean, resSigma) =
                (sc._2.mean(predictionIndex), sc._2.sigma(predictionIndex))

              val (predictedMean, actualval, sigma) =
                (
                  resSigma * pattern._2._1 + resMean,
                  resSigma * pattern._2._2 + resMean,
                  resSigma * (pattern._2._1 - pattern._2._3) / model._errorSigma.toDouble
                )

              val label = if ((actualval - dst_t) <= threshold) 1.0 else 0.0

              val normalDist = Gaussian(predictedMean - dst_t, sigma)
              (normalDist.cdf(threshold), label)

            })
        }
      )

    logger.info("Collating results for Hour t + " + (predictionIndex + 1))
    val flow = preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      DataPipe(
        (testDat: Iterable[(DenseVector[Double], DenseVector[Double])]) =>
          (sc._1 * sc._2)(testDat)
      ) >
      postProcessPipe

    flow("data/omni2_" + testStartDate.getYear + ".csv")
  }

  def generatePredictions(
    model: AbstractGPRegressionModel[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int)],
    sc: (GaussianScaler, GaussianScaler),
    predictionIndex: Int = 3
  ) = {

    val (pF, pT) =
      (math.pow(2, orderFeat).toInt, math.pow(2, orderTarget).toInt)
    val arxOrders =
      if (deltaT.isEmpty) List.fill[Int](exogenousInputs.length + 1)(pF)
      else deltaT

    val (hFeat, _) =
      (haarWaveletFilter(orderFeat), haarWaveletFilter(orderTarget))

    val haarWaveletPipe = IterableDataPipe(
      (featAndTarg: (DenseVector[Double], DenseVector[Double])) =>
        if (useWaveletBasis)
          (
            DenseVector(
              featAndTarg._1.toArray
                .grouped(pF)
                .map(l => hFeat(DenseVector(l)).toArray)
                .reduceLeft((a, b) => a ++ b)
            ),
            featAndTarg._2
          )
        else
          featAndTarg
    )

    val (testStartDate, testEndDate) =
      (
        formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT)
      )

    val (tStampStart, tStampEnd) =
      (testStartDate.getMillis / 1000.0, testEndDate.getMillis / 1000.0)

    val filterData = IterableDataPipe(
      (couple: (Double, Double)) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val filterDataARX = IterableDataPipe(
      (couple: (Double, DenseVector[Double])) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val prepareFeaturesAndOutputs = if (exogenousInputs.isEmpty) {
      extractTimeSeries((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterData >
        deltaOperationMult(arxOrders.head, pT)
    } else {
      extractTimeSeriesVec((year, day, hour) => {
        val dt = dayofYearformat.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
        )
        dt.getMillis / 1000.0
      }) >
        filterDataARX >
        deltaOperationARXMult(arxOrders, pT)
    }

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      haarWaveletPipe >
      DataPipe(
        (testDat: Iterable[(DenseVector[Double], DenseVector[Double])]) =>
          (sc._1 * sc._2)(testDat)
      ) >
      DataPipe(
        (nTestDat: Iterable[(DenseVector[Double], DenseVector[Double])]) => {
          //Generate Predictions for each point
          val preds = model.test(nTestDat)

          preds
            .filter(_._1._2 == predictionIndex)
            .map(c => (c._2, c._3, c._4, c._5))
            .toStream //.head
        }
      ) >
      IterableDataPipe((d: (Double, Double, Double, Double)) => {
        val (scMean, scSigma) =
          (sc._2.mean(predictionIndex), sc._2.sigma(predictionIndex))

        (
          d._1 * scSigma + scMean,
          d._2 * scSigma + scMean,
          d._3 * scSigma + scMean,
          d._4 * scSigma + scMean
        )
      })

    flow("data/omni2_" + testStartDate.getYear + ".csv")
  }

  def test(): MultiRegressionMetrics = {

    val (pF, pT) = (1, math.pow(2, orderTarget).toInt)

    OmniMultiOutputModels.exogenousInputs = List()

    val (testStartDate, testEndDate) =
      (
        formatter.parseDateTime(testStart).minusHours(pF),
        formatter.parseDateTime(testEnd).plusHours(pT)
      )

    val (tStampStart, tStampEnd) =
      (testStartDate.getMillis / 1000.0, testEndDate.getMillis / 1000.0)

    val filterData = IterableDataPipe(
      (couple: (Double, Double)) =>
        couple._1 >= tStampStart && couple._1 <= tStampEnd
    )

    val prepareFeaturesAndOutputs = extractTimeSeries((year, day, hour) => {
      val dt = dayofYearformat.parseDateTime(
        year.toInt.toString + "/" + day.toInt.toString + "/" + hour.toInt.toString
      )
      dt.getMillis / 1000.0
    }) >
      filterData >
      deltaOperationMult(pF, pT)

    val flow = preProcess >
      prepareFeaturesAndOutputs >
      IterableDataPipe(
        (c: (DenseVector[Double], DenseVector[Double])) =>
          (DenseVector.fill[Double](c._2.length)(c._1(c._1.length - 1)), c._2)
      ) >
      DataPipe((d: Iterable[(DenseVector[Double], DenseVector[Double])]) => {
        val sc = d.toList
        new MultiRegressionMetrics(sc, sc.length)
      })

    flow("data/omni2_" + testStartDate.getYear + ".csv")

  }

}

object DstPersistenceMOExperiment {

  val logger = Logger.getLogger(this.getClass)

  var stormAverages: Boolean = false

  def apply(orderT: Int) = {
    OmniMultiOutputModels.orderTarget = orderT

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        IterableDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val eventId   = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          //val minDst = stormMetaFields(5).toDouble

          //val stormCategory = stormMetaFields(6)

          OmniMultiOutputModels.testStart = startDate + "/" + startHour
          OmniMultiOutputModels.testEnd = endDate + "/" + endHour

          logger.info(
            "Testing on Storm: " + OmniMultiOutputModels.testStart + " to " + OmniMultiOutputModels.testEnd
          )

          OmniMultiOutputModels.test()
        }) >
        DataPipe(
          (metrics: Iterable[MultiRegressionMetrics]) =>
            metrics.reduce((m, n) => m ++ n).setName("Persistence Dst; OSA")
        )

    stormsPipe("data/geomagnetic_storms.csv")

  }
}

object DstMOGPExperiment {

  val logger = Logger.getLogger(this.getClass)

  var gridSize                           = 3
  var gridStep                           = 0.2
  var logScale                           = false
  var maxIt                              = 10
  var stormAverages: Boolean             = false
  var onsetClassificationScores: Boolean = false

  def train(
    orderF: Int = 4,
    orderT: Int = 3,
    useWavelets: Boolean = true
  )(kernel: CompositeCovariance[(DenseVector[Double], Int)],
    noise: CompositeCovariance[(DenseVector[Double], Int)]
  ): (
    AbstractGPRegressionModel[Stream[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int)],
    (GaussianScaler, GaussianScaler)
  ) = {

    OmniMultiOutputModels.orderFeat = orderF
    OmniMultiOutputModels.orderTarget = orderT
    OmniMultiOutputModels.useWaveletBasis = useWavelets

    OmniMultiOutputModels.orderFeat = orderF
    OmniMultiOutputModels.orderTarget = orderT

    OmniMultiOutputModels.trainStorms(
      kernel,
      noise,
      gridSize,
      gridStep,
      useLogSc = logScale,
      maxIt
    )
  }

  def test(
    model: AbstractGPRegressionModel[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int)],
    scaler: (GaussianScaler, GaussianScaler)
  ) = {

    val processResults = if (!stormAverages) {
      DataPipe(
        (metrics: Iterable[Iterable[RegressionMetrics]]) =>
          metrics.reduceLeft((m, n) => m.zip(n).map(pair => pair._1 ++ pair._2))
      )
    } else {
      IterableDataPipe(
        (m: Iterable[RegressionMetrics]) => m.map(_.kpi() :/ 63.0)
      ) >
        DataPipe(
          (metrics: Iterable[Iterable[DenseVector[Double]]]) =>
            metrics
              .reduceLeft((m, n) => m.zip(n).map(pair => pair._1 + pair._2))
        )
    }

    val processResultsOnsetClassification =
      DataPipe((metrics: Iterable[Iterable[BinaryClassificationMetrics]]) => {
        metrics.reduceLeft((m, n) => m.zip(n).map(pair => pair._1 ++ pair._2))
      })

    val stormsPipe = onsetClassificationScores match {
      case false =>
        fileToStream >
          replaceWhiteSpaces >
          //DataPipe((st: Iterable[String]) => st.take(43)) >
          IterableDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId   = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            //val minDst = stormMetaFields(5).toDouble

            //val stormCategory = stormMetaFields(6)

            OmniMultiOutputModels.testStart = startDate + "/" + startHour
            OmniMultiOutputModels.testEnd = endDate + "/" + endHour

            logger.info(
              "Testing on Storm: " + OmniMultiOutputModels.testStart + " to " + OmniMultiOutputModels.testEnd
            )

            OmniMultiOutputModels.test(model, scaler)
          }) >
          processResults

      case true =>
        fileToStream >
          replaceWhiteSpaces >
          //DataPipe((st: Iterable[String]) => st.take(43)) >
          IterableDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId   = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            //val minDst = stormMetaFields(5).toDouble

            //val stormCategory = stormMetaFields(6)

            OmniMultiOutputModels.testStart = startDate + "/" + startHour
            OmniMultiOutputModels.testEnd = endDate + "/" + endHour

            logger.info(
              "Testing on Storm: " + OmniMultiOutputModels.testStart + " to " + OmniMultiOutputModels.testEnd
            )

            OmniMultiOutputModels.testOnset(model, scaler)
          }) >
          processResultsOnsetClassification
    }

    stormsPipe("data/geomagnetic_storms.csv")

  }

  def testRegression[
    M <: ContinuousProcessModel[Iterable[
      (DenseVector[Double], DenseVector[Double])
    ], (DenseVector[Double], Int), Double, _]
  ](model: M,
    scaler: (GaussianScaler, GaussianScaler)
  ) = {

    val processResults = if (!stormAverages) {
      DataPipe(
        (metrics: Iterable[Iterable[RegressionMetrics]]) =>
          metrics.reduceLeft((m, n) => m.zip(n).map(pair => pair._1 ++ pair._2))
      )
    } else {
      IterableDataPipe(
        (m: Iterable[RegressionMetrics]) => m.map(_.kpi() :/ 63.0)
      ) >
        DataPipe(
          (metrics: Iterable[Iterable[DenseVector[Double]]]) =>
            metrics
              .reduceLeft((m, n) => m.zip(n).map(pair => pair._1 + pair._2))
        )
    }

    val processResultsOnsetClassification =
      DataPipe((metrics: Iterable[Iterable[BinaryClassificationMetrics]]) => {
        metrics.reduceLeft((m, n) => m.zip(n).map(pair => pair._1 ++ pair._2))
      })

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        //DataPipe((st: Iterable[String]) => st.take(43)) >
        IterableDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val eventId   = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          //val minDst = stormMetaFields(5).toDouble

          //val stormCategory = stormMetaFields(6)

          OmniMultiOutputModels.testStart = startDate + "/" + startHour
          OmniMultiOutputModels.testEnd = endDate + "/" + endHour

          logger.info(
            "Testing on Storm: " + OmniMultiOutputModels.testStart + " to " + OmniMultiOutputModels.testEnd
          )

          OmniMultiOutputModels.test(model, scaler)
        }) >
        processResults

    stormsPipe("data/geomagnetic_storms.csv")

  }

}
