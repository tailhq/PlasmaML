package io.github.mandar2812.PlasmaML

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes.DataPipe3

/**
  * @author mandar2812 date 30/03/2017.
  */
class RadialDiffusionSolver(
  lShellLimits: (Double, Double), timeLimits: (Double, Double),
  nL: Int, nT: Int, processDiffusionParams: DataPipe3[
  DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double],
  Seq[(DenseMatrix[Double], DenseVector[Double])]]) {

  val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

  val stackFactory = NeuralStackFactory(
    (1 to nT).map(_ => new Vec2VecLayerFactory(VectorLinear)(nL+1, nL+1)):_*
  )

  val computeStackParameters = processDiffusionParams

  def getComputationStack = computeStackParameters > stackFactory

  def solve(
    lossProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double])(
    f0: DenseVector[Double]): Seq[DenseVector[Double]] =
    getComputationStack(lossProfile, diffusionProfile, boundaryFlux).forwardPropagate(f0)

}

object RadialDiffusionSolver {


  def getForwardModelParameters(
    lShellLimits: (Double, Double),
    timeLimits: (Double, Double),
    nL: Int, nT: Int)(
    lossProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double]) = {

    val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)



  }
}
