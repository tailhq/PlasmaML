package io.github.mandar2812.PlasmaML.vanAllen

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.PlasmaML.cdf.CDFUtils
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.models.{GLMPipe, GPRegressionPipe}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.models.lm.GeneralizedLinearModel
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.GaussianScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

class CRRESKernel(th: Double, s: Double, a: Double = 0.5, b: Double = 0.5) extends SVMKernel[DenseMatrix[Double]]
with LocalSVMKernel[DenseVector[Double]] {

  val waveletK = new WaveletKernel((x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0))(th)
  //new WaveKernel(th)

  val rbfK = new RBFKernel(s)

  val coefficients = List("rbf", "wavelet")

  override val hyper_parameters = coefficients ++ waveletK.hyper_parameters ++ rbfK.hyper_parameters

  state = waveletK.state ++ rbfK.state ++ Map("rbf" -> math.abs(a), "wavelet" -> math.abs(b))

  /*override def setHyperParameters(h: Map[String, Double]): CRRESKernel.this.type = {
    waveletK.setHyperParameters(h)
    rbfK.setHyperParameters(h)
    super.setHyperParameters(h)
  }*/

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val (x_mlt, y_mlt) = (x(0 until 2), y(0 until 2 ))
    val (x_rest, y_rest) = (x(2 to -1), y(2 to -1))
    state("wavelet")*waveletK.evaluate(x_mlt, y_mlt)+ state("rbf")*rbfK.evaluate(x_rest, y_rest)
  }
}




/**
  * @author mandar2812 date: 26/5/16.
  */
object CRRESTest {

  var columns = Seq(
    "FLUX", "MLT", "L", "Local_Time", "N",
    "B", "Bmin", "Latitude", "Altitude")

  val fillValKey = "FILLVAL"

  def prepareData = CDFUtils.readCDF >
    CDFUtils.cdfToStream(columns) >
    DataPipe((couple: (Stream[Seq[AnyRef]], Map[String, Map[String, String]])) => {
      val (data, metadata) = couple
      data.filter(record => {
        val (mlt, lshell, local_time, b, bmin, lat, alt) = (
          record(1).toString, record(2).toString,
          record(3).toString, record(5).toString,
          record(6).toString, record(7).toString,
          record(8).toString)

        mlt != metadata("MLT")(fillValKey) &&
          lshell != metadata("L")(fillValKey) &&
          b != metadata("B")(fillValKey) &&
          bmin != metadata("Bmin")(fillValKey) &&
          lat != metadata("Latitude")(fillValKey)
      }).map(record => {
        val (flux, mlt, lshell,
        local_time, n, b,
        bmin, lat, alt) = (
          record.head.asInstanceOf[Array[Float]],
          record(1).asInstanceOf[Float],
          record(2).asInstanceOf[Float],
          record(3).asInstanceOf[Float],
          record(4).asInstanceOf[Array[Float]],
          record(5).asInstanceOf[Float],
          record(6).asInstanceOf[Float],
          record(7).asInstanceOf[Float],
          record(8).asInstanceOf[Float])
        val filteredFlux = flux.filter(f => f.toString != metadata("FLUX")(fillValKey)).map(_.toDouble)
        val filteredJ = n.filter(f => f.toString != metadata("N")(fillValKey)).map(_.toDouble)
        (DenseVector(
          mlt.toDouble, lshell.toDouble, b.toDouble,
          bmin.toDouble, lat.toDouble),
          filteredFlux.sum/filteredFlux.length)
      })
    })


  def preProcess(num_train: Int, num_test: Int) =
    prepareData >
    DataPipe((data: Stream[(DenseVector[Double], Double)]) => {
      (data.take(num_train), data.takeRight(num_test))
    }) >
    trainTestGaussianStandardization

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

  def apply(kernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
            new RBFKernel(2.0),
            noiseKernel: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] =
            new DiracKernel(2.0),
            num_train: Int = 500, num_test: Int = 1000,
            grid: Int = 5, step: Double = 0.2) = {

    val gpPipe = new GPRegressionPipe[GPRegression,
      ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))](
      (dataCouple) => dataCouple._1._1,
      kernel, noiseKernel)

    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val m = gpPipe(trainTest)

        val tunePipe = DataPipe((model: GPRegression) => {

          val scalerFeatures = new GaussianScaler(
            trainTest._2._1(0 until trainTest._2._1.length - 1),
            trainTest._2._2(0 until trainTest._2._2.length - 1))

          val scaleTargets = new GaussianScaler(
            DenseVector(trainTest._2._1(-1)),
            DenseVector(trainTest._2._2(-1)))

          val sc = scalerFeatures * scaleTargets

          val validationPipe = prepareData > StreamDataPipe(
            (s: (DenseVector[Double], Double)) => {
              (s._1, DenseVector(s._2))
            }) >
            DataPipe((s: Stream[(DenseVector[Double], DenseVector[Double])]) => sc(s.takeRight(num_test))) >
            StreamDataPipe((s: (DenseVector[Double], DenseVector[Double])) => (s._1, s._2(0)))

          model.validationSet = validationPipe("/var/Datasets/space-weather/crres/crres_h0_mea_19910601_v01.cdf")

          model
        }) > modelTuning(
          kernel.effective_state ++ noiseKernel.effective_state,
          "GS", grid, step)

        val (tunedModel, config) = tunePipe(m)

        tunedModel.setState(config)
        GPRegressionTest(tunedModel)((trainTest._1._2, trainTest._2))
      }

    val finalPipe = preProcess(num_train, num_test) > DataPipe(modelTrainTest)

    finalPipe("/var/Datasets/space-weather/crres/crres_h0_mea_19910201_v01.cdf")
  }

  def apply(hidden: Int, acts: List[String], nCounts: List[Int],
            num_train: Int, num_test: Int, stepSize: Double,
            maxIt: Int, alpha: Double, regularization: Double,
            mini: Double) = {

    val trainTest = prepareData >
      StreamDataPipe(
        (couple: (DenseVector[Double], Double)) =>
          (couple._1, DenseVector(couple._2))) >
      DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        (data.take(num_train), data.takeRight(num_test))
      }) >
      gaussianScalingTrainTest >
      DataPipe((datSc: (Stream[(DenseVector[Double], DenseVector[Double])],
            Stream[(DenseVector[Double], DenseVector[Double])],
            (GaussianScaler, GaussianScaler))) => {

        val gr = FFNeuralGraph(
          datSc._1.head._1.length, 1,
          hidden, acts, nCounts)

        val model = new FeedForwardNetwork(
          datSc._1, gr,
          StreamDataPipe(identity[(DenseVector[Double], DenseVector[Double])] _))

        model.setLearningRate(stepSize)
          .setMaxIterations(maxIt)
          .setBatchFraction(mini)
          .setMomentum(alpha)
          .setRegParam(regularization)
          .learn()

        val reverseScale = datSc._3._2.i * datSc._3._2.i
        val res = reverseScale(model.test(datSc._2)).map(c => (c._1(0), c._2(0))).toList


        val results = new RegressionMetrics(res, res.length)
        results.generateFitPlot()
        results.print()
      })

    trainTest("/var/Datasets/space-weather/crres/crres_h0_mea_19910122_v01.cdf")

  }

  def apply(phi: (DenseVector[Double]) => DenseVector[Double],
            num_train: Int, num_test: Int, stepSize: Double,
            maxIt: Int, regularization: Double,
            mini: Double) = {
    val modelPipe = new GLMPipe[
      (DenseMatrix[Double], DenseVector[Double]),
      Stream[(DenseVector[Double], Double)]](identity _, phi) >
      trainParametricModel[Stream[(DenseVector[Double], Double)],
        DenseVector[Double], DenseVector[Double], Double,
        (DenseMatrix[Double], DenseVector[Double]),
        GeneralizedLinearModel[(DenseMatrix[Double], DenseVector[Double])]](regularization, stepSize, maxIt)

    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val model = modelPipe(trainTest._1._1)

        val scoresAndLabels = trainTest._1._2.map(testpattern => {
          (model.predict(testpattern._1), testpattern._2)
        }).map(scoAndLab => {
          val (score, target) = scoAndLab
          (score * trainTest._2._2(-1) + trainTest._2._2(-1),
            target * trainTest._2._2(-1) + trainTest._2._2(-1))
        }).toList

        val metrics = new RegressionMetrics(scoresAndLabels, scoresAndLabels.length)
        metrics.print()
        metrics.generateFitPlot()
      }

    val finalPipe = preProcess(num_train, num_test) > DataPipe(modelTrainTest)

    finalPipe("/var/Datasets/space-weather/crres/crres_h0_mea_19910201_v01.cdf")
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
