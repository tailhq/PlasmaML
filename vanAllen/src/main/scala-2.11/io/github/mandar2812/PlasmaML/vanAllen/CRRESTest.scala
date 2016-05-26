package io.github.mandar2812.PlasmaML.vanAllen

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.PlasmaML.cdf.CDFUtils
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, DiracKernel, RBFKernel}
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.models.neuralnets.{FFNeuralGraph, FeedForwardNetwork}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DynaMLPipe, GPRegressionPipe}

/**
  * Created by mandar on 26/5/16.
  */
object CRRESTest {

  var columns = Seq("FLUX", "MLT", "L")

  def preProcess(num_train: Int, num_test: Int) = CDFUtils.readCDF >
    CDFUtils.cdfToStream(columns) >
    DataPipe((couple: (Stream[Seq[AnyRef]], Map[String, Map[String, String]])) => {
      val (data, metadata) = couple
      data.filter(record => {
        val (mlt, lshell) = (record(1).toString, record(2).toString)
        mlt != metadata("MLT")("FILLVAL") && lshell != metadata("L")("FILLVAL")
      }).map(record => {
        val (flux, mlt, lshell) = (
          record.head.asInstanceOf[Array[Float]],
          record(1).asInstanceOf[Float],
          record(2).asInstanceOf[Float])
        val filteredFlux = flux.filter(f => f.toString != metadata("FLUX")("FILLVAL")).map(_.toDouble)
        (DenseVector(mlt.toDouble, lshell.toDouble), filteredFlux.sum/filteredFlux.length)
      })
    }) >
    DataPipe((data: Stream[(DenseVector[Double], Double)]) => {
      (data.take(num_train), data.takeRight(num_test))
    }) >
    DynaMLPipe.trainTestGaussianStandardization

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
      DynaMLPipe.modelTuning(
        kernel.state ++ noiseKernel.state,
        "GS", grid, step)


    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val (tunedModel, config) = modelPipe(trainTest)
        tunedModel.setState(config)
        DynaMLPipe.GPRegressionTest(tunedModel)((trainTest._1._2, trainTest._2))
      }

    val finalPipe = preProcess(num_train, num_test) > DataPipe(modelTrainTest)

    finalPipe("/var/Datasets/space-weather/crres/crres_h0_mea_19910201_v01.cdf")
  }

  def apply(hidden: Int, acts: List[String], nCounts: List[Int],
            num_train: Int, num_test: Int, stepSize: Double,
            maxIt: Int, alpha: Double, regularization: Double,
            mini: Double) = {
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val gr = FFNeuralGraph(trainTest._1._1.head._1.length, 1, hidden,
          acts, nCounts)

        val transform = DataPipe((d: Stream[(DenseVector[Double], Double)]) =>
          d.map(el => (el._1, DenseVector(el._2))))

        val model = new FeedForwardNetwork[Stream[(DenseVector[Double], Double)]](trainTest._1._1, gr, transform)

        model.setLearningRate(stepSize)
          .setMaxIterations(maxIt)
          .setBatchFraction(mini)
          .setMomentum(alpha)
          .setRegParam(regularization)
          .learn()

        val res = model.test(trainTest._1._2)
        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], DenseVector[Double])]) =>
              res.map(i => (i._1(0), i._2(0))).toList) > DataPipe((list: List[(Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)

        metrics.print()
        metrics.generatePlots()
      }

    val finalPipe = preProcess(num_train, num_test) > DataPipe(modelTrainTest)

    finalPipe("/var/Datasets/space-weather/crres/crres_h0_mea_19910201_v01.cdf")
  }
}
