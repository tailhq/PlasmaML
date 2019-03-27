package io.github.mandar2812.PlasmaML.streamer

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.utils.GaussianScaler
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.models.stp.StudentTRegression
import io.github.mandar2812.dynaml.optimization.{AbstractCSA, CoupledSimulatedAnnealing, GridSearch}
import io.github.mandar2812.dynaml.pipes._

import scala.util.Random

/**
  * Created by mandar on 19/09/16.
  */
object StreamerShowerEmulator {

  var dataFile: String = "data/all_results.txt"

  var trainingSize = 1000
  var testSize = 1000

  var globalOpt: String = "GS"

  val preprocess = fileToStream >
    dropHead >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      (4 to 11).toList,
      Map()
    ) >
    removeMissingLines >
    DataPipe((data: Iterable[String]) => {
      val size = data.toSeq.length
      val fraction: Double = (trainingSize + testSize)/size.toDouble
      //Sub sample a fraction of the total data set.
      data.filter(_ => Random.nextDouble <= fraction)
    }) >
    IterableDataPipe((line: String) => {
      val split = line.split(",")
      val features = split.take(4).map(_.toDouble).map(math.log)
      val counts = split.takeRight(4).map(_.toDouble).map(c => if(c != 0.0) c else c + 1.0).map(math.log)

      val totalCount = counts.last

      List(10.0, 100.0, 1000.0).zipWithIndex.map((couple) => {
        (DenseVector(features++Array(math.log(couple._1))), counts(couple._2))
      })
    }) >
    DataPipe((d: Iterable[List[(DenseVector[Double], Double)]]) => {
      d.reduce(_++_).toStream
    }) >
    DataPipe((data: Stream[(DenseVector[Double], Double)]) => {
      val size = data.length
      val fraction: Double = (trainingSize + testSize)/size.toDouble
      //Sub sample a fraction of the total data set.
      data.filter(_ => Random.nextDouble <= fraction)
    })

  val generateData =
    StreamDataPipe((pattern: (DenseVector[Double], Double)) =>
      (pattern._1, DenseVector(pattern._2))) >
      DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) => {
        val size = data.length
        (data.take(trainingSize), data.takeRight(size-trainingSize))
      }) >
      gaussianScalingTrainTest

  def apply(kernel: LocalScalarKernel[DenseVector[Double]],
            noise: LocalScalarKernel[DenseVector[Double]],
            grid: Int, step: Double, useLogSc: Boolean, maxIt:Int): RegressionMetrics = {

    val flow = preprocess >
      generateData >
      DataPipe((dataAndScales: (
        Iterable[(DenseVector[Double], DenseVector[Double])],
        Iterable[(DenseVector[Double], DenseVector[Double])],
          (GaussianScaler, GaussianScaler))) => {

        val model = new GPRegression(
          kernel, noise,
          dataAndScales._1.map(p => (p._1, p._2(0))).toSeq)

        val gs = globalOpt match {
          case "CSA" =>
            new CoupledSimulatedAnnealing(model)
              .setGridSize(grid)
              .setStepSize(step)
              .setLogScale(useLogSc)
              .setMaxIterations(maxIt)
              .setVariant(AbstractCSA.MwVC)
          case "GS" =>
            new GridSearch(model)
              .setGridSize(grid)
              .setStepSize(step)
              .setLogScale(useLogSc)
        }

        val startConf = kernel.effective_state ++ noise.effective_state

        val (tunedGP, _) = gs.optimize(startConf)

        val predictions = tunedGP.test(dataAndScales._2.map(p => (p._1, p._2(0))).toSeq)
          .map(p => (DenseVector(p._3), DenseVector(p._2))).toStream

        val scAndL = (dataAndScales._3._2.i*dataAndScales._3._2.i)(predictions).map(c => (c._1(0), c._2(0)))

        new RegressionMetrics(scAndL.toList, scAndL.length)
      })

    flow(dataFile)
  }



  def apply(kernel: LocalScalarKernel[DenseVector[Double]],
            noise: LocalScalarKernel[DenseVector[Double]], mu: Double,
            grid: Int, step: Double, useLogSc: Boolean, maxIt:Int): RegressionMetrics = {

    val flow = preprocess >
      generateData >
      DataPipe((dataAndScales: (
        Iterable[(DenseVector[Double], DenseVector[Double])],
        Iterable[(DenseVector[Double], DenseVector[Double])],
          (GaussianScaler, GaussianScaler))) => {

        val model = new StudentTRegression(
          mu,
          kernel, noise,
          dataAndScales._1.map(p => (p._1, p._2(0))).toSeq)

        val gs = globalOpt match {
          case "CSA" =>
            new CoupledSimulatedAnnealing(model)
              .setGridSize(grid)
              .setStepSize(step)
              .setLogScale(useLogSc)
              .setMaxIterations(maxIt)
              .setVariant(AbstractCSA.MwVC)
          case "GS" =>
            new GridSearch(model)
              .setGridSize(grid)
              .setStepSize(step)
              .setLogScale(useLogSc)
        }

        val startConf = kernel.effective_state ++ noise.effective_state ++ Map("degrees_of_freedom" -> mu)

        val (tunedGP, _) = gs.optimize(startConf)

        val predictions = tunedGP.test(dataAndScales._2.map(p => (p._1, p._2(0))).toSeq)
          .map(p => (DenseVector(p._3), DenseVector(p._2))).toStream

        val scAndL = (dataAndScales._3._2.i*dataAndScales._3._2.i)(predictions).map(c => (c._1(0), c._2(0)))

        new RegressionMetrics(scAndL.toList, scAndL.length)
      })

    flow(dataFile)
  }

}
