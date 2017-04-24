package io.github.mandar2812.PlasmaML.omni

//Scala language imports
//scala mutable collections api
import io.github.mandar2812.dynaml.modelpipe.GLMPipe
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.models.sgp.ESGPModel
import io.github.mandar2812.dynaml.optimization.ProbGPCommMachine

import scala.collection.mutable.{MutableList => MList}

//Logging system
import org.apache.log4j.Logger

//Linear algebra and csv utilities
import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter

//Import Joda time libraries
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

//Import relevant components from DynaML
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.optimization.GridSearch
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.lm.GeneralizedLinearModel
import io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GradBasedGlobalOptimizer}
import io.github.mandar2812.dynaml.pipes.{BifurcationPipe, StreamFlatMapPipe}

/**
  * A consolidation of models and work-flows for OSA prediction
  * and reproducible research on the hourly resolution OMNI data.
  *
  * @author mandar2812 date: 18/01/2017.
  * */
object OmniOSA {

  //Define some types to make code more readable
  type Features = DenseVector[Double]
  type Output = Double

  type Data = Stream[(Features, Output)]
  type Scales = (GaussianScaler, GaussianScaler)
  type DataAndScales = (Data, Scales)

  type VectorKernel = LocalScalarKernel[Features]

  type DateSection = (String, String)
  type TimeStamp = Double

  type GP = AbstractGPRegressionModel[Seq[(Features, Output)], Features]

  type SGP = ESGPModel[Data, Features]

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
  //List of storms not contained in Ji et.al but contained in stormsFile2
  val stormsFile3 = "geomagnetic_storms3.csv"

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
    47 -> "99999.99", 15 -> "999.9")

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

  var standardise: Boolean = false

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

  private var p_ex: List[Int] = List()

  private var modelType: String = "GP-AR"

  def modelType_(m: String) = {
    if(m == "GP-NARMAX") {
      exogenousInputs = List(24, 15, 16, 28)
      targetColumn = 40
      p_target = 2
      p_ex = List(3)
    }
    modelType = m
  }

  /**
    * Returns the model auto-regressive order with respect
    * to the model target as well as the exogenous inputs
    * */
  def modelOrders = (p_target, p_ex)

  def input_dimensions = OmniOSA.modelOrders._1+OmniOSA.modelOrders._2.sum

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
  def setExogenousVars(
    cols: List[Int],
    orders: List[Int],
    changeModelType: Boolean = true): Unit = {

    /*require(
      cols.length == orders.length,
      "Number of exogenous variables must be equal to number of elements in 'orders' list")*/

    require(
      cols.forall(c => c >= 0 && c < 55),
      "All elements of 'cols' list must contain column indices "+
      "which are actually present in OMNI hourly resolution data")
    require(orders.forall(_>0), "Auto-regressive orders for model exogenous variables must be positive")

    //Done with sanity checks now for assignments
    exogenousInputs = cols
    p_ex = orders
    if(changeModelType)
      modelType = if(cols.isEmpty) "GP-AR" else "GP-ARX"
  }


  def clearExogenousVars() = setExogenousVars(List(), List())

  //Define a variable which stores the training data sections
  //specified as a list of string 2-tuple defining the
  //start and end of each section.
  var trainingDataSections: Stream[DateSection] = Stream(
    ("2010/01/03/00", "2010/01/11/23"),
    ("2011/08/05/20", "2011/08/06/22"))

  //Set the validation data sections to empty
  var validationDataSections: Stream[DateSection] = Stream.empty[DateSection]

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

  val extractNarmaxFeatures = StreamDataPipe((couple: (TimeStamp, Features)) => {
    val features = couple._2
    //Calculate the coupling function p^0.5 V^4/3 Bt sin^6(theta)
    val Bt = math.sqrt(math.pow(features(2), 2) + math.pow(features(3), 2))
    val sin_theta6 = math.pow(features(2)/Bt, 6)
    val p = features(4)
    val v = features(1)
    val couplingFunc = math.sqrt(p)*math.pow(v, 4/3.0)*Bt*sin_theta6
    val Dst = features(0)
    (couple._1, DenseVector(Dst, couplingFunc))
  })

  /**
    * Returns a data pipeline which takes a stream of the data
    * in its string form and converts it to a usable time lagged
    * data set for training an AR model.
    * */
  def prepareData(start: String, end: String): DataPipe[Stream[String], Data] = {

    val hoursOffset = if(exogenousInputs.isEmpty) p_target else math.max(p_target, p_ex.max)

    val (startStamp, endStamp) = (
      formatter.parseDateTime(start).minusHours(hoursOffset).getMillis/1000.0,
      formatter.parseDateTime(end).getMillis/1000.0)

    //Return the data pipe which creates a time lagged
    //version of the data received by it and filters it
    //according to the time window supplied in the arguments
    //i.e. start and end

    modelType match {

      //If we are training NARMAX inspired GP model
      case "GP-NARMAX" =>
        extractTimeSeriesVec((year,day,hour) => {
          dayofYearformatter.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString +
              "/" + hour.toInt.toString).getMillis/1000.0 }) >
          StreamDataPipe((couple: (Double, Features)) =>
            couple._1 >= startStamp && couple._1 <= endStamp) >
          extractNarmaxFeatures >
          deltaOperationARX(List(p_target)++p_ex)

      //If we are training a AR model
      case "GP-AR" =>
        extractTimeSeries((year,day,hour) => {
          dayofYearformatter.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString +
              "/" + hour.toInt.toString)
            .getMillis/1000.0 }) >
          StreamDataPipe((couple: (Double, Double)) =>
            couple._1 >= startStamp && couple._1 <= endStamp) >
          deltaOperation(p_target, 0)

      //Else assume we are training an ARX model
      case _ =>
        extractTimeSeriesVec((year,day,hour) => {
          dayofYearformatter.parseDateTime(
            year.toInt.toString + "/" + day.toInt.toString +
              "/" + hour.toInt.toString).getMillis/1000.0 }) >
          StreamDataPipe((couple: (Double, Features)) =>
            couple._1 >= startStamp && couple._1 <= endStamp) >
          deltaOperationARX(List(p_target)++p_ex)
    }
  }

  def processTimeSegment = DataPipe((dateLimits: DateSection) => {
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
    StreamDataPipe(processTimeSegment) >
    DataPipe((segments: Stream[Data]) => {
      segments.foldLeft(Stream.empty[(Features, Output)])((segA, segB) => segA ++ segB)
    })

  //Pipelines for conversion of targets to and fro vector to double
  val preNormalisation =
    StreamDataPipe((record: (Features, Output)) => (record._1, DenseVector(record._2)))

  val postNormalisation =
    StreamDataPipe((record: (Features, Features)) => (record._1, record._2(0)))


  //Define the global optimization parameters
  var globalOpt: String = "GS"
  var gridSize: Int = 3
  var gridStep: Double = 0.2
  var useLogScale: Boolean = false
  //Iterations only required for ML or CSA based global optimization
  var maxIterations: Int = 20

  /**
    * Returns a pipeline which returns the prediction of the Persistence model.
    * E[f(t)] = f(t-1).
    * */
  def meanFuncPersistence = DataPipe((features: Features) => features(p_target-1))

  val meanFuncNarmax = DataPipe((vec: Features) => {
    val Dst_t_1 = vec(1)
    val Dst_t_2 = vec(0)

    val couplingFunc_t_1 = vec(4)
    val couplingFunc_t_2 = vec(3)
    val couplingFunc_t_3 = vec(2)

    val finalFeatures = DenseVector(Dst_t_1, couplingFunc_t_1, couplingFunc_t_1*Dst_t_1,
      Dst_t_2, math.pow(couplingFunc_t_2, 2.0),
      couplingFunc_t_3, math.pow(couplingFunc_t_1, 2.0),
      couplingFunc_t_2, 1.0, math.pow(Dst_t_1, 2.0))


    Narmax(finalFeatures)
  })

  /**
    * The complete data pipeline from raw OMNI data
    * to standardized attributes and targets.
    * */
  def dataPipeline = compileSegments >
    preNormalisation >
    calculateGaussianScales(standardise) >
    DataPipe(
      postNormalisation,
      identityPipe[(GaussianScaler, GaussianScaler)]
    )

  val getStormTimeRanges = DataPipe((stormEventData: String) => {
    val stormMetaFields = stormEventData.split(',')
    val startDate = stormMetaFields(1)
    val startHour = stormMetaFields(2).take(2)

    val endDate = stormMetaFields(3)
    val endHour = stormMetaFields(4).take(2)

    (startDate+"/"+startHour, endDate+"/"+endHour)
  })

  /**
    * Returns a pipeline which takes as input a file containing a list
    * of storm events and converts them into a stream of tuples containing
    * the start and end of each event.
    *
    * @param num_storms The number of storms to choose (from the end of the file)
    *
    * */
  def extractValidationSectionsPipe(num_storms: Int = 10) = fileToStream >
    replaceWhiteSpaces >
    StreamDataPipe(getStormTimeRanges) >
    DataPipe((s: Stream[DateSection]) => s.takeRight(num_storms))

  def validationDataPipeline(scales: (GaussianScaler, GaussianScaler)) = compileSegments

  /**
    * Train a parametric model which
    * can be used as a mean function for
    * GP/STP models
    * */
  def trainMeanFunc(reg: Double = 0.1) = {

    type OptData = (DenseMatrix[Double], Features)

    val flowpre = dataPipeline > DataPipe((data: DataAndScales) => data._1)
    val pipe =
      new GLMPipe[
        OptData,
        Stream[DateSection]](flowpre.run) >
        trainParametricModel[
          Data, Features, Features, Output, OptData,
          GeneralizedLinearModel[OptData]](reg)

    pipe(Stream(("2008/01/01/00", "2008/12/31/23")))
  }

  /**
    * A pipeline which takes data and its associated scales
    * (scales are represented by some subclass of [[io.github.mandar2812.dynaml.pipes.ReversibleScaler]])
    *
    * */
  def gpTrain(
    kernel: VectorKernel,
    noise: VectorKernel,
    meanFunc: DataPipe[Features, Output] = null) =
    DataPipe((dataAndScales: DataAndScales) => {
      val trainingData = dataAndScales._1
      implicit val ev = VectorField(dataAndScales._1.head._1.length)


      val kSc = CovarianceFunction(dataAndScales._2._1)
      val targetSampleVariance = math.pow(dataAndScales._2._2(0).sigma, 2.0)

      val meanF =
        if (meanFunc == null) DataPipe((_:Features) => dataAndScales._2._2(0).mean)
        else meanFunc

      val model = new GPRegression(
        (kSc>kernel)*targetSampleVariance, (kSc>noise)*targetSampleVariance,
        trainingData, meanF)

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

        case "ML" =>
          new GradBasedGlobalOptimizer(model).setStepSize(gridStep)

        case "GPC" =>
          new ProbGPCommMachine(model)
            .setGridSize(gridSize)
            .setStepSize(gridStep)
            .setLogScale(useLogScale)
            .setMaxIterations(maxIterations)
      }

      val startConfig = kernel.effective_state ++ noise.effective_state

      val (tunedModel, c) = modelTuner.optimize(
        startConfig,
        Map(
          "persist" -> "true",
          "tolerance" -> "0.0001",
          "step" -> gridStep.toString,
          "maxIterations" -> maxIterations.toString))

      //tunedModel.setState(c)
      (tunedModel, dataAndScales._2)
    })

  def sgpTrain(
    kernel: VectorKernel,
    noise: VectorKernel,
    lambda: Double, tau: Double,
    meanFunc: DataPipe[Features, Output] = null) =
    DataPipe((dataAndScales: DataAndScales) => {
      val trainingData = dataAndScales._1
      implicit val ev = VectorField(dataAndScales._1.head._1.length)
      implicit val transform = DataPipe((data: Data) => data.toSeq)

      val kSc = CovarianceFunction(dataAndScales._2._1)
      val targetSampleVariance = math.pow(dataAndScales._2._2(0).sigma, 2.0)

      val meanF =
        if (meanFunc == null) DataPipe((_:Features) => dataAndScales._2._2(0).mean)
        else meanFunc

      val model: SGP = ESGPModel(
        (kSc> kernel)*targetSampleVariance, (kSc>noise)*targetSampleVariance,
        meanF, lambda, tau)(trainingData)

      val modelTuner = globalOpt match {
        case "GS" =>
          new GridSearch[model.type](model)
            .setGridSize(gridSize)
            .setStepSize(gridStep)
            .setLogScale(useLogScale)

        case "CSA" =>
          new CoupledSimulatedAnnealing[model.type](model)
            .setGridSize(gridSize)
            .setStepSize(gridStep)
            .setLogScale(useLogScale)
            .setMaxIterations(maxIterations)
            .setVariant(CoupledSimulatedAnnealing.MwVC)

      }

      val startConfig = kernel.effective_state ++ noise.effective_state ++ Map("skewness" -> lambda, "cutoff" -> tau)

      val (tunedModel, c) = modelTuner.optimize(
        startConfig,
        Map(
          "persist" -> "true",
          "tolerance" -> "0.0001",
          "step" -> gridStep.toString,
          "maxIterations" -> maxIterations.toString))

      (tunedModel, dataAndScales._2)
    })

  def sgpTest(testFile: String = stormsFileJi) =
    DataPipe((modelAndScales: (SGP, Scales)) => {
      //Break task into two pipelines

      //pipeline1 is the usual transformations for
      //reading the data from the correct time ranges
      //and extracting features and targets from them
      val pipeline1 =
      fileToStream >
        replaceWhiteSpaces >
        StreamFlatMapPipe(getStormTimeRanges > processTimeSegment)

      //Pipeline2 takes the storm time data and performs predictions
      //and generates a RegressionMetrics object
      val pipeline2 =
      DataPipe((testData: Data) =>
        modelAndScales._1
          .test(testData).map(t => (DenseVector(t._3), DenseVector(t._2)))
          .toStream) >
        StreamDataPipe((c: (Features, Features)) => (c._1(0), c._2(0))) >
        DataPipe((results: Stream[(Double, Double)]) =>
          new RegressionMetrics(results.toList, results.length)
            .setName(modelType+" "+columnNames(targetColumn)+"; OSA")
        )

      val stormFilePipeline = pipeline1 > pipeline2
      stormFilePipeline(dataDir+testFile)
    })


  /**
    * Returns a pipeline which takes a model and the data scaling and
    * tests it on a list of geomagnetic storms supplied in the parameter
    *
    * */
  def gpTest(testFile: String = stormsFileJi) =
    DataPipe((modelAndScales: (GP, Scales)) => {
      //Break task into two pipelines

      //pipeline1 is the usual transformations for
      //reading the data from the correct time ranges
      //and extracting features and targets from them
      val pipeline1 =
        fileToStream >
        replaceWhiteSpaces >
        StreamFlatMapPipe(getStormTimeRanges > processTimeSegment)

      //Pipeline2 takes the storm time data and performs predictions
      //and generates a RegressionMetrics object
      val pipeline2 =
        DataPipe((testData: Data) =>
          modelAndScales._1
            .test(testData).map(t => (DenseVector(t._3), DenseVector(t._2)))
            .toStream) >
        StreamDataPipe((c: (Features, Features)) => (c._1(0), c._2(0))) >
        DataPipe((results: Stream[(Double, Double)]) =>
          new RegressionMetrics(results.toList, results.length)
            .setName(modelType+" "+columnNames(targetColumn)+"; OSA")
        )

      val stormFilePipeline = pipeline1 > pipeline2
      stormFilePipeline(dataDir+testFile)
    })

  def generateGPPredictions(testFile: String = stormsFileJi) =
    DataPipe((modelAndScales: (GP, Scales)) => {

      val fileNamePrefix = testFile.split(".csv").head

      val (model, scales) = modelAndScales

      val pipeline1 =
        fileToStream >
          replaceWhiteSpaces >
          StreamDataPipe(getStormTimeRanges > processTimeSegment)

      val pipeline2 =
        DataPipe((storms: Stream[Data]) => {
          //For each storm generate predictions
          //and output them in the appropriate file

          storms.zipWithIndex.foreach(stormCouple => {
            val stormName = "storm"+(stormCouple._2+1).toString
            logger.info("Generating Predictions for Storm: "+(stormCouple._2+1))
            val stormData = stormCouple._1
            //Generate Predictions
            val stormPredictions =
              model.test(stormData)
                .map(preds => Seq(preds._2, preds._3, preds._4, preds._5))
                .toStream
            //Write values to a file dump
            logger.info("Dumping predictions to csv: "+(stormCouple._2+1))
            valuesToFile(
              dataDir+"/"+fileNamePrefix+"_"+stormName+".csv")(
              stormPredictions)
          })
        })

      val predictionsPipeline = pipeline1 > pipeline2

      predictionsPipeline(dataDir+testFile)
    })

  def generateSGPPredictions(testFile: String = stormsFileJi) =
    DataPipe((modelAndScales: (SGP, Scales)) => {

      val fileNamePrefix = testFile.split(".csv").head

      val (model, scales) = modelAndScales

      val pipeline1 =
        fileToStream >
          replaceWhiteSpaces >
          StreamDataPipe(getStormTimeRanges > processTimeSegment)

      val pipeline2 =
        DataPipe((storms: Stream[Data]) => {
          //For each storm generate predictions
          //and output them in the appropriate file

          storms.zipWithIndex.foreach(stormCouple => {
            val stormName = "storm"+(stormCouple._2+1).toString
            logger.info("Generating Predictions for Storm: "+(stormCouple._2+1))
            val stormData = stormCouple._1
            //Generate Predictions
            val stormPredictions =
              model.test(stormData)
                .map(preds => Seq(preds._2, preds._3, preds._4, preds._5))
                .toStream
            //Write values to a file dump
            logger.info("Dumping predictions to csv: "+(stormCouple._2+1))
            valuesToFile(
              dataDir+"/"+fileNamePrefix+"_sgp"+"_"+stormName+".csv")(
              stormPredictions)
          })
        })

      val predictionsPipeline = pipeline1 > pipeline2

      predictionsPipeline(dataDir+testFile)
    })


  /**
    * Train a Gaussian Process model on the specified training sections i.e. [[trainingDataSections]]
    *
    * @param kernel The covariance function as a [[LocalScalarKernel]] instance
    * @param noise The noise as a [[LocalScalarKernel]] instance
    * */
  def buildGPOnTrainingSections(
    kernel: VectorKernel, noise: VectorKernel,
    meanFunc: DataPipe[Features, Output] = DataPipe((_:Features) => 0.0)): (GP, Scales) = {

    val pipeline = dataPipeline > gpTrain(kernel, noise, meanFunc)

    pipeline(trainingDataSections)
  }

  def buildAndTestGP(
    kernel: VectorKernel,
    noise: VectorKernel,
    meanFunc: DataPipe[Features, Double] = DataPipe((_:Features) => 0.0),
    stormFile: String = stormsFileJi) = {

    val pipeline =
      dataPipeline >
      gpTrain(kernel, noise, meanFunc) >
      gpTest(stormsFileJi)

    pipeline(trainingDataSections)

  }

  def buildAndTestSGP(
    kernel: VectorKernel,
    noise: VectorKernel,
    lambda: Double, tau: Double,
    meanFunc: DataPipe[Features, Double] = DataPipe((_:Features) => 0.0),
    stormFile: String = stormsFileJi) = {

    val pipeline =
      dataPipeline >
        sgpTrain(kernel, noise, lambda, tau, meanFunc) >
        sgpTest(stormsFileJi)

    pipeline(trainingDataSections)

  }

  /**
    * Carry out large scale model comparison experiment.
    * on GP-AR or GP-ARX models.
    * */
  def experiment(
    kernel: VectorKernel,
    noise: VectorKernel,
    meanFunc: DataPipe[Features, Double] = DataPipe((_:Features) => 0.0),
    orders: Range = 1 to 10,
    modelSelectionStorms: String = stormsFile3,
    testStorms: String = stormsFileJi,
    resultsFile: String = "OmniOSARes.csv",
    zeros: List[Int] = List.fill[Int](2)(0)) = {

    noise.block_all_hyper_parameters
    val originalState = kernel.state ++ noise.state

    val file = new java.io.File(dataDir+resultsFile)
    val fileOpened: Boolean = !file.exists() || file.length() == 0

    val writer = CSVWriter.open(new java.io.File(dataDir+resultsFile), append = true)

    val columnNames = Seq("model", "modelSize","order") ++
      (1 to 2).map(i => "order_ex_"+i) ++
      Seq("data") ++ originalState.keys.toSeq ++
      Seq("globalOpt", "gridSize", "step", "maxIt") ++
      Seq("mae", "rmse", "cc", "rsq")

    val globalOptConfig =
      globalOpt match {
        case "GS" =>
          Seq(globalOpt, gridSize, gridStep, 0)
        case "CSA" =>
          Seq(globalOpt, gridSize, gridStep, maxIterations)
        case "ML" =>
          Seq("ML", 0, gridStep, maxIterations)
      }


    //If the file has already been written to then write the header row.
    if (fileOpened)
      writer.writeRow(columnNames)

    orders.foreach(ord => {

      //Create a list of model orders which sum to ord
      //For a GP-AR model it is ord, 0, 0
      //for a GP-ARX model it is ord1, ord2, ord3; st ord1+ord2+ord3 = ord
      val listOfModelOrders: List[(Int, List[Int])] =
        if(modelType != "GP-AR") {
          for(order_dst <- 1 to ord;
              order_v <- 1 to ord;
              order_b <- 1 to ord)
            yield (order_dst, List(order_v, order_b))
        }.filter(o => o._1+o._2.sum == ord).toList
      else List((ord, List(0, 0)))

      logger.info("_______________________________")
      logger.info("Running experiment for model order: "+ord)
      logger.info("Possible model orders: \n"+listOfModelOrders+"\n")
      logger.info("_______________________________")

      //Loop over the list of possible model orders
      listOfModelOrders.foreach(modelOrders => {
        val (orderTarget, order_ex) = modelOrders

        //Set the target variable to Dst and set
        //its autoregressive order to ordTarget
        setTarget(40, orderTarget)


        if(modelType == "GP-ARX") {
          setExogenousVars(List(24, 16), order_ex)
        } else if(modelType == "GP-AR") {
          clearExogenousVars()
        }

        //Reset the kernel
        kernel.setHyperParameters(originalState)
        //Create a model train/tune pipeline
        val pipeline = dataPipeline > gpTrain(kernel, noise, meanFunc)

        try {

          //Apply the pipeline on the training data sections
          val (model, scalers) = pipeline(trainingDataSections)

          //Create a pipeline to test the model on
          //two different sets of storms
          val scoresPipe = BifurcationPipe(
            gpTest(modelSelectionStorms),
            gpTest(testStorms))

          //Calculate scores on validation and test sets.
          val (validationPerformance, testPerformance) = scoresPipe((model, scalers))

          //Obtain metrics for model
          val (valP, tP) = (
            Seq(
              validationPerformance.mae, validationPerformance.rmse,
              validationPerformance.corr, validationPerformance.Rsq),
            Seq(
              testPerformance.mae, testPerformance.rmse,
              testPerformance.corr, testPerformance.Rsq))

          //Initialize some of the data to be dumped into the result csv
          val lineCommonData = Seq(modelType, model.npoints, orderTarget) ++ order_ex
          val selectedState = originalState.keys.map(k => model._current_state(k))

          //Write results to file
          writer.writeAll(Seq(
            lineCommonData ++ Seq("validation") ++ selectedState ++ globalOptConfig ++ valP,
            lineCommonData ++ Seq("test") ++ selectedState ++ globalOptConfig ++ tP
          ))
        } catch {
          case e: Exception =>
            logger.info(e.toString)
        }
      })
    })

    writer.close()
  }

}
