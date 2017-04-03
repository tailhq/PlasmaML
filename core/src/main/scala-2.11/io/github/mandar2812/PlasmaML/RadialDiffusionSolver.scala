package io.github.mandar2812.PlasmaML

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes.DataPipe3

/**
  * @author mandar2812 date 30/03/2017.
  *
  * Implementation of a discrete radial diffusion system.
  *
  */
class RadialDiffusionSolver(
  lShellLimits: (Double, Double),
  timeLimits: (Double, Double),
  nL: Int, nT: Int) {

  val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

  val stackFactory = NeuralStackFactory(
    (1 to nT).map(_ => new Vec2VecLayerFactory(VectorLinear)(nL+1, nL+1)):_*
  )

  val computeStackParameters =
    DataPipe3(RadialDiffusionSolver.getForwardModelParameters(lShellLimits, timeLimits, nL, nT))

  def getComputationStack = computeStackParameters > stackFactory

  def solve(
    lossProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double])(
    f0: DenseVector[Double]): Seq[DenseVector[Double]] =
    getComputationStack(lossProfile, diffusionProfile, boundaryFlux).forwardPropagate(f0)

}

object RadialDiffusionSolver {


  def forwardConvDiff(i: Int, j: Int)(n: Int, m: Int): Double =
    if(n - i >= 0 && n - i <= 1 && m - j >= 0 && m - j <= 1) 0.25 else 0.0

  def backwardConvDiff(i: Int, j: Int)(n: Int, m: Int): Double =
    if(i - n >= 0 && i - n <= 1 && m - j >= 0 && m - j <= 1) 0.25 else 0.0

  def forwardConvBoundaryFlux(j:Int, i: Int)(n: Int, m: Int): Double =
    if(m == i && n == j) -1.0 else if(m == i+1 && n == j) 1.0 else 0.0

  def forwardConvLossProfile(i: Int, j: Int)(n: Int, m: Int): Double =
    if(n == i && m - j >= 0 && m - j <= 1) 0.5 else 0.0

  def conv(filter: (Int, Int) => Double)(data: DenseMatrix[Double]) =
    sum(data.mapPairs((coords, value) => value * filter(coords._1, coords._2)))

  def getForwardModelParameters(
    lShellLimits: (Double, Double),
    timeLimits: (Double, Double),
    nL: Int, nT: Int)(
    lossProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double]): Seq[(DenseMatrix[Double], DenseVector[Double])] = {

    val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

    val lVec = DenseVector.tabulate[Double](nL + 1)(i =>
      if(i < nL) lShellLimits._1+(deltaL*i) else lShellLimits._2)
      .map(v => 0.5*v*v/deltaL)

    val alpha: (Int) => DenseMatrix[Double] = (n) => {

      DenseMatrix.tabulate[Double](nL + 1, nL + 1)((j, k) => {
        if (math.abs(j-k) > 1) {
          0.0
        } else if(j == k) {
          if(j == 0 || j == nL + 1) {
            1.0
          } else {
            1/deltaT - 0.5*conv(forwardConvLossProfile(j,n))(lossProfile) -
              lVec(j)*(conv(forwardConvDiff(j, n))(diffusionProfile) + conv(backwardConvDiff(j, n))(diffusionProfile))
          }
        } else if(j > k) {
          lVec(j)*conv(backwardConvDiff(j, n))(diffusionProfile)
        } else {
          lVec(j)*conv(forwardConvDiff(j, n))(diffusionProfile)
        }

      })
    }

    val beta: (Int) => DenseMatrix[Double] = (n) => {
      DenseMatrix.tabulate[Double](nL + 1, nL + 1)((j, k) => {
        if (math.abs(j-k) > 1) {
          0.0
        } else if(j == k) {
          if(j == 0 || j == nL + 1) {
            1.0
          } else {
            1/deltaT + 0.5*conv(forwardConvLossProfile(j,n))(lossProfile) +
              lVec(j)*(conv(forwardConvDiff(j, n))(diffusionProfile) + conv(backwardConvDiff(j, n))(diffusionProfile))
          }
        } else if(j > k) {
          -lVec(j)*conv(backwardConvDiff(j, n))(diffusionProfile)
        } else {
          -lVec(j)*conv(forwardConvDiff(j, n))(diffusionProfile)
        }

      })
    }

    val gamma: (Int) => DenseVector[Double] = (n) => {
      DenseVector.tabulate[Double](nL + 1)(l =>
        if( l == 0 || l == nL) conv(forwardConvBoundaryFlux(l,n))(boundaryFlux) else 0.0
      )
    }

    (0 until nT).map(n => {
      (beta(n)\alpha(n), beta(n)\gamma(n))
    })
  }
}
