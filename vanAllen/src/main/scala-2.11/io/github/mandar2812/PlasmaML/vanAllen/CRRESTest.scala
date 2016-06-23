package io.github.mandar2812.PlasmaML.vanAllen

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.PlasmaML.cdf.CDFUtils
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.models.{GLMPipe, GPRegressionPipe}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, DiracKernel, RBFKernel}
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.models.lm.GeneralizedLinearModel
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.GaussianScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

/**
  * @author mandar2812 date: 26/5/16.
  */
object CRRESTest {

  var columns = Seq("FLUX", "MLT", "L", "Local_Time", "N", "B", "Bmin")

  def prepareData = CDFUtils.readCDF >
    CDFUtils.cdfToStream(columns) >
    DataPipe((couple: (Stream[Seq[AnyRef]], Map[String, Map[String, String]])) => {
      val (data, metadata) = couple
      data.filter(record => {
        val (mlt, lshell, local_time, b, bmin) = (
          record(1).toString, record(2).toString,
          record(3).toString, record(5).toString,
          record(6).toString)

        mlt != metadata("MLT")("FILLVAL") &&
          lshell != metadata("L")("FILLVAL") &&
          b != metadata("B")("FILLVAL") &&
          bmin != metadata("Bmin")("FILLVAL")
      }).map(record => {
        val (flux, mlt, lshell, local_time, n, b, bmin) = (
          record.head.asInstanceOf[Array[Float]],
          record(1).asInstanceOf[Float],
          record(2).asInstanceOf[Float],
          record(3).asInstanceOf[Float],
          record(4).asInstanceOf[Array[Float]],
          record(5).asInstanceOf[Float],
          record(6).asInstanceOf[Float])
        val filteredFlux = flux.filter(f => f.toString != metadata("FLUX")("FILLVAL")).map(_.toDouble)
        val filteredJ = n.filter(f => f.toString != metadata("N")("FILLVAL")).map(_.toDouble)
        (DenseVector(
          mlt.toDouble, lshell.toDouble, b.toDouble,
          bmin.toDouble),
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
        mlt != metadata("MLT")("FILLVAL") &&
          lshell != metadata("L")("FILLVAL") //&&
        // local_time != metadata("Local_Time")("FILLVAL")
      }).map(record => {
        val (flux, mlt, lshell, local_time) = (
          record.head.asInstanceOf[Array[Float]],
          record(1).asInstanceOf[Float],
          record(2).asInstanceOf[Float],
          record(3).asInstanceOf[Float])
        val filteredFlux = flux.filter(f => f.toString != metadata("FLUX")("FILLVAL")).map(_.toDouble)
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

    val modelPipe = new GPRegressionPipe[GPRegression,
      ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))](
      (dataCouple) => dataCouple._1._1,
      kernel, noiseKernel) >
      modelTuning(
        kernel.state ++ noiseKernel.state,
        "GS", grid, step)

    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val (tunedModel, config) = modelPipe(trainTest)
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
