package io.github.mandar2812.PlasmaML.vanAllen

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import io.github.mandar2812.PlasmaML.cdf.{CDFUtils, EpochFormatter}
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.models.{GLMPipe, GPRegressionPipe}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.models.lm.{GeneralizedLinearModel, RegularizedGLM}
import io.github.mandar2812.dynaml.models.neuralnets.{AutoEncoder, FeedForwardNetwork}
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions._
import io.github.mandar2812.dynaml.pipes.{StreamDataPipe, _}
import io.github.mandar2812.dynaml.utils.{GaussianScaler, MinMaxScaler}
import io.github.mandar2812.dynaml.utils._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import com.quantifind.charts.Highcharts._
import org.joda.time.{DateTime, DateTimeZone}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

import scala.util.Random


class CRRESKernel(th: Double, s: Double, a: Double = 0.5, b: Double = 0.5) extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]] {

  val waveletK = new WaveletKernel((x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0))(th)

  val rbfK = new RBFKernel(s)

  val coefficients = List("rbf", "wavelet")

  override val hyper_parameters = coefficients ++ waveletK.hyper_parameters ++ rbfK.hyper_parameters

  state = waveletK.state ++ rbfK.state ++ Map("rbf" -> math.abs(a), "wavelet" -> math.abs(b))

  override def setHyperParameters(h: Map[String, Double]): CRRESKernel.this.type = {
    waveletK.setHyperParameters(h)
    rbfK.setHyperParameters(h)
    super.setHyperParameters(h)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val (x_mlt, y_mlt) = (x(0 until 2), y(0 until 2 ))
    val (x_rest, y_rest) = (x(2 to -1), y(2 to -1))
    state("wavelet")*waveletK.evaluate(x_mlt, y_mlt) + state("rbf")*rbfK.evaluate(x_rest, y_rest)
  }
}




/**
  * @author mandar2812 date: 26/5/16.
  */
object CRRESTest {

  DateTimeZone.setDefault(DateTimeZone.UTC)

  val dateTimeFormatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss.SSS")

  val epochFormatter = new EpochFormatter()

  var columns = Seq(
    "FLUX", "MLT", "L", "Latitude", "Epoch", "energy", "angle")

  var columnDataTypes = Seq()

  val fillValKey = "FILLVAL"

  var dataRoot = "/var/Datasets/space-weather/"

  val crresScale = DataPipe((trainTest: (Stream[(DenseVector[Double], DenseVector[Double])],
    Stream[(DenseVector[Double], DenseVector[Double])])) => {

    val (num_features, num_targets) = (trainTest._1.head._1.length, trainTest._1.head._2.length)

    val (min, max) = getMinMax(trainTest._1.map(tup =>
      DenseVector(tup._1.toArray ++ tup._2.toArray)).toList)

    val featuresScaler = new MinMaxScaler(min(0 until num_features), max(0 until num_features))

    val targetsScaler = new MinMaxScaler(
      min(num_features until num_features + num_targets),
      max(num_features until num_features + num_targets))

    val scaler: ReversibleScaler[(DenseVector[Double], DenseVector[Double])] = featuresScaler * targetsScaler

    (scaler(trainTest._1), scaler(trainTest._2),
      (featuresScaler.asInstanceOf[ReversibleScaler[DenseVector[Double]]],
        targetsScaler.asInstanceOf[ReversibleScaler[DenseVector[Double]]]))
  })

  var fileIds = Stream("19900728")

  var testFileIds = Stream("19910303")

  def traintestFile = dataRoot+"crres/crres_h0_mea_19910601_v01.cdf"

  def validationFile = dataRoot+"crres/crres_h0_mea_19910601_v01.cdf"

  var (featuresSc, targetsSc): (GaussianScaler, GaussianScaler) = (null, null)

  object OptConfig {
    var stepSize = 1.0
    var reg = 0.00
    var maxIt = 100
    var momentum = 0.5
    var sparsity = 0.2
    var mini = 1.0
  }

  def processCRRESCDF = CDFUtils.readCDF >
    CDFUtils.cdfToStream(columns) >
    DataPipe((couple: (Stream[Seq[AnyRef]], Map[String, Map[String, String]])) => {
      val (data, metadata) = couple
      (data.filter(record => {
        val (mlt, lshell, lat) = (
          record(1).toString,
          record(2).toString,
          record(3).toString)

        mlt != metadata("MLT")(fillValKey) &&
          lshell != metadata("L")(fillValKey) &&
          lat != metadata("Latitude")(fillValKey)
      }), metadata)
    })


  def preProcess(num_train: Int, num_test: Int) =
    collateData >
      StreamDataPipe((s: (DenseVector[Double], Double)) => (s._1, DenseVector(math.log(s._2)))) >
      DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        (data.take(num_train), data.takeRight(num_test))
      })

  def precalculateScales(): Unit = {
    val process = collateData >
      StreamDataPipe((s: (DenseVector[Double], Double)) => (s._1, DenseVector(math.log(s._2)))) >
      gaussianScaling

    //(featuresSc, targetsSc) = process(fileIds)._2
  }

  def preProcessRDD() = CDFUtils.readCDF >
    CDFUtils.cdfToRDD(columns) >
    DataPipe((couple: (RDD[Seq[AnyRef]], Map[String, Map[String, String]])) => {
      val (data, metadata) = couple
      data.filter(record => {
        val (mlt, lshell, local_time) = (record(1).toString, record(2).toString, record(3).toString)
        mlt != metadata("MLT")(fillValKey) &&
          lshell != metadata("L")(fillValKey) //&&
        // local_time != metadata("Local_Time")(fillValKey)
      }).map(record => {
        val (flux, mlt, lshell, local_time) = (
          record.head.asInstanceOf[Array[Float]],
          record(1).asInstanceOf[Float],
          record(2).asInstanceOf[Float],
          record(3).asInstanceOf[Float])
        val filteredFlux = flux.filter(f => f.toString != metadata("FLUX")(fillValKey)).map(_.toDouble)
        (DenseVector(mlt.toDouble, lshell.toDouble),
          filteredFlux.sum/filteredFlux.length)
      })
    })

  val extractTimeSeriesSymh = (Tfunc: (Double, Double, Double, Double) => Long) =>
    DataPipe((lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = Tfunc(splits(0).toDouble, splits(1).toDouble, splits(2).toDouble, splits(3).toDouble)
      val feat = DenseVector(splits.slice(4, splits.length).map(_.toDouble))
      (timestamp, feat)
    })

  val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H/m")

  val prepCRRES = processCRRESCDF >
    DataPipe((couple: (Stream[Seq[AnyRef]], Map[String, Map[String, String]])) => {
      val (data, metadata) = couple
      data.map(record => {
        val (flux, mlt, lshell, lat, epoch) = (
          record.head.asInstanceOf[Array[Float]],
          record(1).asInstanceOf[Float],
          record(2).asInstanceOf[Float],
          record(3).asInstanceOf[Float],
          record(4).asInstanceOf[Double].toLong)

        val filteredFlux = flux.filter(f => f.toString != metadata("FLUX")(fillValKey)).map(_.toDouble)

        val dt = new DateTime(epochFormatter.formatEpoch(epoch)).minuteOfHour().roundFloorCopy()

        (DenseVector(
          mlt.toDouble, lshell.toDouble,
          lat.toDouble, dt.getMillis.toDouble/1000.0),
          filteredFlux.sum/filteredFlux.length)
      })
    }) >
    StreamDataPipe((p: (DenseVector[Double], Double)) => p._1(1) <= 7.0 && p._1(1) >= 3.0) >
    StreamDataPipe((p: (DenseVector[Double], Double)) =>
      (p._1(p._1.length-1).toLong, (p._1(0 to -2), p._2)))

  val prepPipeSymH = fileToStream >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      List(0,1,2,3,40,41,42,43),
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
    extractTimeSeriesSymh((year,day,hour, minute) => {
      val dt = dayofYearformat.parseDateTime(
        year.toInt.toString + "/" + day.toInt.toString + "/" +
          hour.toInt.toString + "/" + minute.toInt.toString)

      (dt.getMillis/1000.0).toLong
    })

  val collateData = StreamDataPipe((s: String) =>
    (dataRoot+"crres/crres_h0_mea_"+s+"_v01.cdf", "data/omni_min"+s.take(4)+".csv")) >
    StreamDataPipe((s: (String, String)) => {
      val (crresData, symhData) = (prepCRRES * prepPipeSymH)(s)

      (crresData ++ symhData).groupBy(_._1).mapValues({
        case Stream((key1, value1), (key2, value2)) =>
          val symh = value2.asInstanceOf[DenseVector[Double]]
          val crresD = value1.asInstanceOf[(DenseVector[Double], Double)]
          (DenseVector(crresD._1.toArray ++ symh.toArray), crresD._2)
        case _ =>
          (DenseVector(0.0), Double.NaN)
      }).values
        .filter(l => l._2 != Double.NaN && l._1.length > 2)
        .toStream
    }) >
    DataPipe((seq: Stream[Stream[(DenseVector[Double], Double)]]) => seq.reduce((x,y) => x ++ y))

  def apply(kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
            new RBFKernel(2.0),
            noiseKernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
            new DiracKernel(2.0),
            num_train: Int = 500, num_test: Int = 1000,
            grid: Int = 5, step: Double = 0.2) = {

    val gpPipe = new GPRegressionPipe[GPRegression,
      (Stream[(DenseVector[Double], DenseVector[Double])],
        Stream[(DenseVector[Double], DenseVector[Double])],
        (ReversibleScaler[DenseVector[Double]], ReversibleScaler[DenseVector[Double]]))](
      (dataCouple) => dataCouple._1.take(num_train).map(c => (c._1, c._2(0))),
      kernel, noiseKernel)

    val buildModel = DataPipe((trainTest: (
      Stream[(DenseVector[Double], DenseVector[Double])],
      Stream[(DenseVector[Double], DenseVector[Double])],
      (GaussianScaler, GaussianScaler))) => {

      val tunePipe = gpPipe > modelTuning[GPRegression](
        kernel.effective_state ++ noiseKernel.effective_state,
        "GS", grid, step)

      (tunePipe(trainTest)._1, trainTest._2, trainTest._3)
    })

    val modelTest = DataPipe(
      (trainTest: (GPRegression,
        Stream[(DenseVector[Double], DenseVector[Double])],
        (GaussianScaler, GaussianScaler))) => {

        val scaleTargets = trainTest._3._2

        val sc: ReversibleScaler[(DenseVector[Double], DenseVector[Double])] = scaleTargets * scaleTargets

        val tunedModel = trainTest._1

        val predRes = tunedModel.test(trainTest._2.map(c => (c._1, c._2(0))))

        val scoresAndLabelsPipe = DataPipe(
          (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
            res.map(i => (DenseVector(i._3), DenseVector(i._2))).toStream) >
          StreamDataPipe((l: (DenseVector[Double], DenseVector[Double])) => sc.i(l)) >
          StreamDataPipe((l: (DenseVector[Double], DenseVector[Double])) => (l._1(0), l._2(0)))

        val scoresAndLabels = scoresAndLabelsPipe(predRes)

        new RegressionMetrics(scoresAndLabels.toList, scoresAndLabels.length)
      })

    val prepare = collateData >
      StreamDataPipe((s: (DenseVector[Double], Double)) => (s._1, DenseVector(math.log(s._2)))) >
      DataPipe((s: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        val shuffledData = Random.shuffle(s)
        val len = shuffledData.length
        (shuffledData.take(len-num_test), shuffledData.takeRight(num_test))
      })


    val workflow = prepare > gaussianScalingTrainTest > buildModel > modelTest

    workflow(fileIds ++ testFileIds)
  }

  /*def apply(hidden: Int, acts: List[String], nCounts: List[Int],
            num_train: Int, num_test: Int) = {

    val trainTest = collateData >
      StreamDataPipe((s: (DenseVector[Double], Double)) => (s._1, math.log(s._2))) >
      StreamDataPipe(
        (couple: (DenseVector[Double], Double)) =>
          (couple._1, DenseVector(couple._2))) >
      DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        (data.take(num_train), data.takeRight(num_test))
      }) >
      crresScale >
      DataPipe((datSc: (Stream[(DenseVector[Double], DenseVector[Double])],
        Stream[(DenseVector[Double], DenseVector[Double])],
        (ReversibleScaler[DenseVector[Double]], ReversibleScaler[DenseVector[Double]]))) => {

        val gr = FFNeuralGraph(
          datSc._1.head._1.length, 1,
          hidden, acts, nCounts)

        val model = new FeedForwardNetwork(
          datSc._1, gr,
          StreamDataPipe(identity[(DenseVector[Double], DenseVector[Double])] _))

        model.setLearningRate(OptConfig.stepSize)
          .setMaxIterations(OptConfig.maxIt)
          .setBatchFraction(OptConfig.mini)
          .setMomentum(OptConfig.momentum)
          .setRegParam(OptConfig.reg)
          .learn()

        val reverseScale = datSc._3._2.i * datSc._3._2.i
        val res = reverseScale(model.test(datSc._2)).map(c => (c._1(0), c._2(0))).toList


        val results = new RegressionMetrics(res, res.length)
        //results.generateFitPlot()
        results.print()
      })

    trainTest(Stream("19910112"))

  }*/
  
  //Test AutoEncoder Idea
  def apply(numExtractedFeatures: Int,
            num_train: Int,
            num_test: Int) = {

    val finalPipe = collateData >
      StreamDataPipe((s: (DenseVector[Double], Double)) => (s._1, math.log(s._2))) >
      StreamDataPipe(
        (couple: (DenseVector[Double], Double)) =>
          (couple._1, DenseVector(couple._2))) >
      DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        (data.take(num_train), data.takeRight(num_test))
      }) >
      crresScale >
      DataPipe((tt: (Stream[(DenseVector[Double], DenseVector[Double])],
        Stream[(DenseVector[Double], DenseVector[Double])],
        (ReversibleScaler[DenseVector[Double]], ReversibleScaler[DenseVector[Double]]))) => {
        //train the autoencoder
        val autoEncoder = new AutoEncoder(
          tt._1.head._1.length,
          numExtractedFeatures,
          List(SIGMOID, LIN))

        autoEncoder.optimizer
          .setMomentum(OptConfig.momentum)
          .setNumIterations(OptConfig.maxIt)
          .setStepSize(OptConfig.stepSize)
          .setRegParam(0.0)
          .setSparsityWeight(OptConfig.sparsity)

        autoEncoder.learn(tt._1.map(p => (p._1, p._1)))

        //transform inputs for train and test

        val new_tt = (
          tt._1.map(pattern => (autoEncoder(pattern._1), pattern._2)),
          tt._2.map(pattern => (autoEncoder(pattern._1), pattern._2))
          )

        val glm = new RegularizedGLM(
          new_tt._1.map(t => (t._1, t._2(0))),
          new_tt._1.length,
          identity[DenseVector[Double]] _)

        glm.setState(Map("regularization" -> OptConfig.reg))

        glm.learn()

        val res = new_tt._2.map(pattern =>
          (tt._3._2.i(DenseVector(glm.predict(pattern._1)))(0), tt._3._2.i(pattern._2)(0)))
          .toList

        val metrics = new RegressionMetrics(res, res.length)
        metrics.generateFitPlot()
        metrics.print()
      })

    finalPipe(Stream("19910112"))
  }


  //Test Sheeleys model
  def apply(num_test: Int, file: String) = {

    val pipe = prepCRRES >
      StreamDataPipe((p: (Long, (DenseVector[Double], Double))) => p._2) >
      DataPipe((s: Stream[(DenseVector[Double], Double)]) => s.takeRight(num_test)) >
      StreamDataPipe((p: (DenseVector[Double], Double)) => {
        val (mlt, lshell, flux) = (p._1(0), p._1(1), p._2)
        val n_threshold = 10*math.pow(lshell/6.6, 4.0)
        flux >= n_threshold match {
          case true =>
            (1390*math.pow(3.0/lshell, 4.83), flux)
          case false =>
            (124*math.pow(3.0/lshell, 4.0) +
              30*math.pow(3.0/lshell, 3.5)*math.cos((mlt - (7.7*math.pow(3.0/lshell, 2.0) + 12))*math.Pi/12.0),
              flux)
        }

      }) >
      DataPipe((s: Stream[(Double, Double)]) => {
        new RegressionMetrics(s.map(c => (math.log(c._1), math.log(c._2))).toList, s.length)
      })

    pipe(file)

  }

  def getAvgMEAFluxHist(logFlag: Boolean = true) = {

    val mapF = (rec: (DenseVector[Double], Double)) => if(logFlag) math.log(rec._2) else rec._2

    /*val pipe = processCRRESCDF >
      StreamDataPipe(mapF) >
      DataPipe((s: Stream[Double]) => {
        histogram(s.toList)
      })

    pipe(traintestFile)*/

    val cumulativeDist = (s: Stream[Double]) => (x: Double) => {
      s.count(_ <= x).toDouble/s.length.toDouble
    }

    val prepPipe = prepCRRES >
      StreamDataPipe((p: (Long, (DenseVector[Double], Double))) => p._2._2)

    val compositePipe =
      StreamDataPipe((s: String) => dataRoot+"crres/crres_h0_mea_"+s+"_v01.cdf") >
        StreamDataPipe((s: String) => prepPipe(s)) >
        DataPipe((seq: Stream[Stream[Double]]) => seq.reduce((x,y) => x ++ y)) >
        DataPipe((xs: Stream[Double]) => {
          val F = cumulativeDist(xs)
          xs.map(v => (math.log(v), math.log(-1.0 * math.log(1.0 - F(v)))))
        }) >
        StreamDataPipe((c: (Double, Double)) => c._2 != Double.PositiveInfinity)

    val fileStrs = Stream(
      "19900728", "19900807", "19900817",
      "19910112", "19910201", "19910122")

    regression(compositePipe(fileStrs))
    xAxis("log(n_average)")
    yAxis("log(-log(1-F(n_average)))")

  }

  /*def apply(committeeSize: Int, fraction: Double) = {
    // From the original RDD construct pipes for subsampling
    // and give them as inputs to a LSSVM Committee
    val rddPipe = preProcessRDD() >
      DataPipe((data: RDD[(DenseVector[Double], Double)]) => {
        val colStats = Statistics.colStats(data.map(pattern =>
          Vectors.dense(pattern._1.toArray ++ Array(pattern._2))))

        val pre = (d: RDD[(DenseVector[Double], Double)]) =>
          d.sample(fraction = fraction, withReplacement = true).collect().toStream

        val pipes = (1 to committeeSize).map(i => {
          val kernel = new RBFKernel(1.0)
          new DLSSVMPipe[RDD[(DenseVector[Double], Double)]](pre, kernel)
        })

      })
  }*/

}

object CRRESExp {

  //def apply
}

object CRRESpsdModels {

  def apply(kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
            new RBFKernel(2.0),
            noiseKernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
            new DiracKernel(2.0),
            num_train: Int = 500, num_test: Int = 1000,
            grid: Int = 5, step: Double = 0.2) = {


    val fileProcess = fileToStream >
      dropHead >
      extractTrainingFeatures(
        List(12,3,4,6,8,5),
        Map(
          12 -> "", 3 -> "",
          4 -> "", 5 -> "",
          9 -> "", 6 -> "" ,
          10 -> "")
      ) >
      removeMissingLines >
      splitFeaturesAndTargets >
      StreamDataPipe((c: (DenseVector[Double], Double)) => (log(c._1), DenseVector(math.log(c._2)))) >
      StreamDataPipe((c: (DenseVector[Double], DenseVector[Double])) => c._2(0) >= -15.0) >
      DataPipe((s: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        val shuffledData = Random.shuffle(s)
        (shuffledData.take(200000), shuffledData.takeRight(num_test))
        }) >
      gaussianScalingTrainTest

    val gpPipe = new GPRegressionPipe[GPRegression,
      (Stream[(DenseVector[Double], DenseVector[Double])],
        Stream[(DenseVector[Double], DenseVector[Double])],
        (ReversibleScaler[DenseVector[Double]], ReversibleScaler[DenseVector[Double]]))](
      (dataCouple) => dataCouple._1.take(num_train).map(c => (c._1, c._2(0))),
      kernel, noiseKernel)

    val buildModel = DataPipe((trainTest: (
      Stream[(DenseVector[Double], DenseVector[Double])],
        Stream[(DenseVector[Double], DenseVector[Double])],
        (GaussianScaler, GaussianScaler))) => {

      val tunePipe = gpPipe > modelTuning[GPRegression](
        kernel.effective_state ++ noiseKernel.effective_state,
        "GS", grid, step)

      (tunePipe(trainTest)._1, trainTest._2, trainTest._3)
    })

    val modelTest = DataPipe(
      (trainTest: (GPRegression,
        Stream[(DenseVector[Double], DenseVector[Double])],
        (GaussianScaler, GaussianScaler))) => {

        val scaleTargets = trainTest._3._2

        val sc: ReversibleScaler[(DenseVector[Double], DenseVector[Double])] = scaleTargets * scaleTargets

        val tunedModel = trainTest._1

        val predRes = tunedModel.test(trainTest._2.map(c => (c._1, c._2(0))))

        val scoresAndLabelsPipe = DataPipe(
          (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
            res.map(i => (DenseVector(i._3), DenseVector(i._2))).toStream) >
          StreamDataPipe((l: (DenseVector[Double], DenseVector[Double])) => sc.i(l)) >
          StreamDataPipe((l: (DenseVector[Double], DenseVector[Double])) => (l._1(0), l._2(0)))

        val scoresAndLabels = scoresAndLabelsPipe(predRes)

        new RegressionMetrics(scoresAndLabels.toList, scoresAndLabels.length)
      })

    val flow = fileProcess > buildModel > modelTest

    flow(CRRESTest.dataRoot+"crres/database_M_fixed_100.csv")


  }
}
