package io.github.mandar2812.PlasmaML.omni

//Scala language imports
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GradBasedGlobalOptimizer}

import scala.collection.mutable.{MutableList => MList}
//Import Joda time libraries
import breeze.linalg.DenseVector
import org.apache.log4j.Logger
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}
//Import relevant compenents from DynaML
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.optimization.GridSearch
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.models.gp.GPRegression

/**
  * @author mandar2812 date: 18/01/2017.
  *
  * A consolidation of models and workflows for OSA prediction
  * and reproducible research on the hourly resolution OMNI data.
  */
object OmniOSA {

  //Set time zone to UTC.
  DateTimeZone.setDefault(DateTimeZone.UTC)

  //Use a joda time parser to convert date-time strings to time stamps
  //Formatter for reading strings consisting of Year/Month/Date/Hour
  val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")
  //Formatter for reading strings consisting of Year/Day of Year (1-365)/Hour
  val dayofYearformatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/D/H")

  //The column indices corresponding to the year, day of year and hour respectively
  val dateColumns = List(0, 1, 2)

  //Initialise the logging system
  private val logger = Logger.getLogger(this.getClass)

  //Store session data in a variable which can be dumped
  val sessionData: MList[Map[String, AnyVal]] = MList()

  //Define location and name of data files
  var dataDir = "data/"
  //List of geomagnetic storms from 1998-2006
  val stormsFileJi = "geomagnetic_storms.csv"
  //List of geomagnetic storms from 1995-2014
  val stormsFile2 = "geomagnetic_storms2.csv"

  /**
    * Stores the missing value strings for
    * each column index in the hourly resolution
    * OMNI files.
    * */
  var columnFillValues = Map(
    16 -> "999.9", 21 -> "999.9",
    24 -> "9999.", 23 -> "999.9",
    40 -> "99999", 22 -> "9999999.",
    25 -> "999.9", 28 -> "99.99",
    27 -> "9.999", 39 -> "999",
    45 -> "99999.99", 46 -> "99999.99",
    47 -> "99999.99")

  /**
    * Contains the name of the quantity stored
    * by its column index (which starts from 0).
    * */
  var columnNames = Map(
    24 -> "Solar Wind Speed",
    16 -> "I.M.F Bz",
    40 -> "Dst",
    41 -> "AE",
    38 -> "Kp",
    39 -> "Sunspot Number",
    28 -> "Plasma Flow Pressure")

  /**
    * Flag which indicates if the term
    * V*Bz should be included in the input
    * features.
    * */
  private var useVBz: Boolean = false

  /**
    * Get the value of the V*Bz flag.
    * */
  def _useVBz: Boolean =
    if ((!exogenousInputs.contains(24) || !exogenousInputs.contains(16)) && useVBz) {
      useVBz = false
      useVBz
    } else {
      useVBz
    }

  /**
    * Set the value of the V*Bz flag, which indicates
    * if the term V*Bz should be included in the input
    * features of the predictive model.
    *
    * @param v The value of the V*Bz flag
    * */
  def useVBz_(v: Boolean): Unit = {
    if (!v)
      useVBz = v
    else if (exogenousInputs.contains(24) && exogenousInputs.contains(16))
      useVBz = v
    else
      logger.warn(
        "Attempt to set V*Bz flag to true "+
          "but V and/or Bz are not included in the exogenousInputs variable")
      logger.warn("Proceeding with existing value (flag) of V*Bz flag.")
  }

  /**
    * Target column indicates which column index
    * is to be treated as the quantity to be predicted
    * in an one step ahead (OSA) fashion. Defaults to 40
    * which is the hourly Dst index.
    * */
  private var targetColumn: Int = 40

  /**
    * Column indices of the exogenous inputs, if any.
    * Defaults to an empty list which implies AR models
    * will be constructed for prediction.
    * */
  private var exogenousInputs: List[Int] = List()

  //Now define variables which store the auto-regressive orders
  //of the models for targets and exogenous inputs.
  private var p_target: Int = 6

  private var p_ex: List[Int] = List.fill[Int](exogenousInputs.length)(1)

  private var modelType: String = "AR"

  /**
    * Returns the model auto-regressive order with respect
    * to the model target as well as the exogenous inputs
    * */
  def modelOrders = (p_target, p_ex)

  /**
    * Set the target variable and the model order
    * with respect to it.
    * */
  def setTarget(col: Int, order: Int): Unit = {
    require(col > 0 && col < 55, "Target column index must actually exist in OMNI hourly resolution data")
    targetColumn = col
    p_target = order
  }

  /**
    * Set the exogenous variables as well as their auto-regressive orders.
    * */
  def setExogenousVars(cols: List[Int], orders: List[Int]): Unit = {
    require(
      cols.length == orders.length,
      "Number of exogenous variables must be equal to number of elements in 'orders' list")
    require(
      cols.forall(c => c >= 0 && c < 55),
      "All elements of 'cols' list must contain column indices "+
      "which are actually present in OMNI hourly resolution data")
    require(orders.forall(_>0), "Auto-regressive orders for model exogenous variables must be positive")

    //Done with sanity checks now for assignments
    exogenousInputs = cols
    p_ex = orders
    modelType = if(cols.isEmpty) "AR" else "ARX"
  }


  def clearExogenousVars() = setExogenousVars(List(), List())

  //Define a variable which stores the training data sections
  //specified as a list of string tuples defining the
  //start and end of each section.
  var trainingDataSections: Stream[(String, String)] = Stream(
    ("2008/01/01/00", "2008/01/11/10"),
    ("2011/08/05/20", "2011/08/06/22"))

  /**
    * Returns a [[io.github.mandar2812.dynaml.pipes.DataPipe]] which
    * reads an OMNI file cleans it and extracts the columns specified
    * by [[targetColumn]] and [[exogenousInputs]].
    * */
  def omniFileToStream = fileToStream >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      dateColumns++List(targetColumn)++exogenousInputs,
      columnFillValues) >
    removeMissingLines


  /**
    * Returns a data pipeline which takes a stream of the data
    * in its string form and converts it to a usable time lagged
    * data set for training an AR model.
    * */
  def prepareData(start: String, end: String):
  DataPipe[Stream[String], Stream[(DenseVector[Double], Double)]] = {

    val hoursOffset = if(exogenousInputs.isEmpty) p_target else math.max(p_target, p_ex.max)
    val (startStamp, endStamp) = (
      formatter.parseDateTime(start).minusHours(hoursOffset).getMillis/1000.0,
      formatter.parseDateTime(end).getMillis/1000.0)

    //Return the data pipe which creates a time lagged
    //version of the data received by it and filters it
    //according to the time window supplied in the arguments
    //i.e. start and end

    modelType match {
      case "AR" =>
        extractTimeSeries((year,day,hour) => {
          dayofYearformatter.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString +
              "/" + hour.toInt.toString).getMillis/1000.0 }) >
          StreamDataPipe((couple: (Double, Double)) =>
            couple._1 >= startStamp && couple._1 <= endStamp) >
          deltaOperation(p_target, 0)
      case _ =>
        extractTimeSeriesVec((year,day,hour) => {
          dayofYearformatter.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString +
              "/" + hour.toInt.toString).getMillis/1000.0 }) >
          StreamDataPipe((couple: (Double, DenseVector[Double])) =>
            couple._1 >= startStamp && couple._1 <= endStamp) >
          deltaOperationARX(List(p_target)++p_ex)
    }
  }

  def processSegment = DataPipe((dateLimits: (String, String)) => {
    //For the given date limits generate the relevant data
    //The file name to read from
    val fileName = dataDir+"omni2_"+dateLimits._1.split("/").head+".csv"
    //Set up a processing pipeline for the file
    val processFile = omniFileToStream > prepareData(dateLimits._1, dateLimits._2)

    //Apply the pipeline to the file
    processFile(fileName)
  })

  /**
    * Returns a pipeline which can be applied on a list of
    * time periods such as [[trainingDataSections]]. The pipeline
    * processes each segment then collates them.
    * */
  def compileSegments =
    StreamDataPipe(processSegment) >
    DataPipe((segments: Stream[Stream[(DenseVector[Double], Double)]]) => {
      segments.foldLeft(Stream.empty[(DenseVector[Double], Double)])((segA, segB) => segA ++ segB)
    })

  //Pipelines for conversion of targets to and fro vector to double
  val preNormalisation =
    StreamDataPipe((record: (DenseVector[Double], Double)) => (record._1, DenseVector(record._2)))

  val postNormalisation =
    StreamDataPipe((record: (DenseVector[Double], DenseVector[Double])) => (record._1, record._2(0)))


  //Define the global optimization parameters
  var globalOpt: String = "GS"
  var gridSize: Int = 3
  var gridStep: Double = 0.2
  var useLogScale: Boolean = false
  //Iterations only required for ML-II or CSA based global optimization
  var maxIterations: Int = 20

  def modelTrain(kernel: LocalScalarKernel[DenseVector[Double]],
                 noise: LocalScalarKernel[DenseVector[Double]]) = DataPipe((dataAndScales: (
    Stream[(DenseVector[Double], Double)],
      (GaussianScaler, GaussianScaler))) => {
    val trainingData = dataAndScales._1
    val model = new GPRegression(kernel, noise, trainingData)
    val modelTuner = globalOpt match {
      case "GS" =>
        new GridSearch[GPRegression](model)
          .setGridSize(gridSize)
          .setStepSize(gridStep)
          .setLogScale(useLogScale)
      case "CSA" =>
        new CoupledSimulatedAnnealing[GPRegression](model)
          .setGridSize(gridSize)
          .setStepSize(gridStep)
          .setLogScale(useLogScale)
          .setMaxIterations(maxIterations)
          .setVariant(CoupledSimulatedAnnealing.MwVC)
      case "ML-II" =>
        new GradBasedGlobalOptimizer(model).setStepSize(gridStep)
    }

    val startConfig = kernel.effective_state ++ noise.effective_state

    val (tunedModel, config) = modelTuner.optimize(startConfig)

    (tunedModel, dataAndScales._2)
  })

  /**
    * Returns a pipeline which takes a model and the data scaling and
    * tests it on a list of geomagnetic storms supplied in the parameter
    *
    * */
  def modelTest(testFile: String = stormsFileJi) =
    DataPipe((modelAndScales: (GPRegression, (GaussianScaler, GaussianScaler))) => {
      val stormFilePipeline = fileToStream >
        replaceWhiteSpaces >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val eventId = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          val minDst = stormMetaFields(5).toDouble

          val stormCategory = stormMetaFields(6)

          //Create a pipeline to process each storm event
          //Start with the usual cleaning
          val stormPipe = processSegment >
            preNormalisation >
            StreamDataPipe(modelAndScales._2._1 * modelAndScales._2._2) >
            postNormalisation >
            //Send storm data to the model for testing
            DataPipe((testData: Stream[(DenseVector[Double], Double)]) =>
              modelAndScales._1
                .test(testData).map(t => (DenseVector(t._3), DenseVector(t._2)))
                .toStream
            ) >
            //Rescale the predicted and actual targets
            StreamDataPipe(modelAndScales._2._2.i * modelAndScales._2._2.i) >
            StreamDataPipe((c: (DenseVector[Double], DenseVector[Double])) => (c._1(0), c._2(0))) >
            //Dump results to a regression metrics object
            DataPipe((results: Stream[(Double, Double)]) =>
              new RegressionMetrics(results.toList, results.length)
                .setName(columnNames(targetColumn)+": OSA")
            )

          stormPipe((startDate+"/"+startHour, endDate+"/"+endHour))
        }) >
        DataPipe((metrics: Stream[RegressionMetrics]) => metrics.reduceLeft((m,n) => m++n))

      stormFilePipeline(dataDir+testFile)
    })


  /**
    * The complete data pipeline from raw OMNI data
    * to standardized attributes and targets.
    * */
  def dataPipeline = compileSegments >
      preNormalisation >
      gaussianScaling >
      DataPipe(
        postNormalisation,
        identityPipe[(GaussianScaler, GaussianScaler)]
      )

  /**
    * Train a Gaussian Process model on the specified training sections i.e. [[trainingDataSections]]
    *
    * @param kernel The covariance function as a [[LocalScalarKernel]] instance
    * @param noise The noise as a [[LocalScalarKernel]] instance
    * */
  def buildGPOnTrainingSections(
    kernel: LocalScalarKernel[DenseVector[Double]],
    noise: LocalScalarKernel[DenseVector[Double]]): (GPRegression, (GaussianScaler, GaussianScaler)) = {

    val pipeline = dataPipeline > modelTrain(kernel, noise)

    pipeline(trainingDataSections)
  }

  def buildAndTestGP(kernel: LocalScalarKernel[DenseVector[Double]],
                     noise: LocalScalarKernel[DenseVector[Double]],
                     stormFile: String = stormsFileJi) = {

    val pipeline = dataPipeline > modelTrain(kernel, noise) > modelTest(stormsFileJi)

    pipeline(trainingDataSections)

  }


}
