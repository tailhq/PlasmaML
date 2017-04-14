package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import io.github.mandar2812.dynaml.algebra.square
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
    (1 to nT).map(_ => new NeuralLayerFactory(RadialDiffusionLayer.metaCompute, VectorLinear)):_*
  )

  val computeStackParameters =
    DataPipe3(RadialDiffusionSolver.getForwardModelParameters(lShellLimits, timeLimits, nL, nT))

  def getComputationStack = computeStackParameters > stackFactory

  def solve(
    lossProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double])(
    f0: DenseVector[Double]): Seq[DenseVector[Double]] =
    getComputationStack(lossProfile, diffusionProfile, boundaryFlux) forwardPropagate f0

}

object RadialDiffusionSolver {

  /**
    * The forward difference convolution filter
    * */
  def forwardConvDiff(i: Int, j: Int)(n: Int, m: Int): Double =
    if(n - i >= 0 && n - i <= 1 && m - j >= 0 && m - j <= 1) 0.25 else 0.0

  /**
    * The backward difference convolution filter
    * */
  def backwardConvDiff(i: Int, j: Int)(n: Int, m: Int): Double =
    if(i - n >= 0 && i - n <= 1 && m - j >= 0 && m - j <= 1) 0.25 else 0.0

  /**
    * Forward difference convolution filter for boundary flux
    * */
  def forwardConvBoundaryFlux(j:Int, i: Int)(n: Int, m: Int): Double =
    if(m == i && n == j) -1.0 else if(m == i+1 && n == j) 1.0 else 0.0

  /**
    * Forward difference convolution filter for loss profiles/scales
    * */
  def forwardConvLossProfile(i: Int, j: Int)(n: Int, m: Int): Double =
    if(n == i && m - j >= 0 && m - j <= 1) 0.5 else 0.0

  def conv(filter: (Int, Int) => Double)(data: DenseMatrix[Double]) =
    sum(data.mapPairs((coords, value) => value * filter(coords._1, coords._2)))

  /**
    *
    * @param lShellLimits The lower and upper limits of L-Shell
    * @param timeLimits The lower and upper limits of time.
    * @param nL The number of partitions to create in the L-Shell domain.
    * @param nT The number of partitions to create in the time domain.
    * @return A sequence of parameters which specify the [[NeuralStack]]
    *         used to compute the radial diffusion solution.
    * */
  def getForwardModelParameters(
    lShellLimits: (Double, Double),
    timeLimits: (Double, Double),
    nL: Int, nT: Int)(
    lossProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double]): Seq[(Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double])] = {

    val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

    val lSqVec = square(DenseVector.tabulate[Double](nL + 1)(i =>
      if(i < nL) lShellLimits._1+(deltaL*i) else lShellLimits._2))

    val adjLVec = lSqVec.map(v => 0.5*v/(deltaL*deltaL))

    val adjustedDiffusionProfile = diffusionProfile.mapPairs((coords, value) => value/lSqVec(coords._1))

    val invDeltaT = 1/deltaT

    /*
    val deltaTMat = DenseMatrix.tabulate[Double](nL + 1, nL + 1)((j, k) => {
      if(j == k) {
        if(j == 0 || j == nL) 1.0 else 1/deltaT
      } else {
        0.0
      }
    })

    val paramsMat: (Int) => DenseMatrix[Double] = (n) => {
      DenseMatrix.tabulate[Double](nL + 1, nL + 1)((j, k) => {
        if (math.abs(j-k) > 1 || (j == 0 && k != j) || (j == nL && k != j)) {
          0.0
        } else if(j == k) {
          if(j == 0 || j == nL) {
            0.0
          } else {
            0.5*conv(forwardConvLossProfile(j,n))(lossProfile) +
              adjLVec(j)*(conv(forwardConvDiff(j, n))(adjustedDiffusionProfile) +
                conv(backwardConvDiff(j, n))(adjustedDiffusionProfile))
          }
        } else if(j > k) {
          -adjLVec(j)*conv(backwardConvDiff(j, n))(adjustedDiffusionProfile)
        } else {
          -adjLVec(j)*conv(forwardConvDiff(j, n))(adjustedDiffusionProfile)
        }

      })
    }
    */

    val paramsTMat: (Int) => (Seq[Seq[Double]], Seq[Seq[Double]]) = (n) => {

      val (alph, bet) = (1 until nL).map(j => {
        val b = 0.5*conv(forwardConvLossProfile(j,n))(lossProfile) +
          adjLVec(j)*(conv(forwardConvDiff(j, n))(adjustedDiffusionProfile) +
            conv(backwardConvDiff(j, n))(adjustedDiffusionProfile))

        val a = -adjLVec(j)*conv(backwardConvDiff(j, n))(adjustedDiffusionProfile)

        val c = -adjLVec(j)*conv(forwardConvDiff(j, n))(adjustedDiffusionProfile)

        (Seq(-a, invDeltaT - b, -c), Seq(a, invDeltaT + b, c))
      }).unzip

      (
        Seq(Seq(0.0, 1.0, 0.0)) ++ alph ++ Seq(Seq(0.0, 1.0, 0.0)),
        Seq(Seq(0.0, 1.0, 0.0)) ++ bet ++ Seq(Seq(0.0, 1.0, 0.0)))
    }

    /*
    * Boundary term gamma(n)
    * */
    val gamma: (Int) => DenseVector[Double] = (n) => {
      DenseVector.tabulate[Double](nL + 1)(l =>
        if( l == 0 || l == nL) conv(forwardConvBoundaryFlux(l,n))(boundaryFlux) else 0.0
      )
    }

    /*
    * Instantiate layer transformations
    * */
    (0 until nT).map(n => {

      /*
      val pM = paramsMat(n)
      val a = deltaTMat - pM
      val b = deltaTMat + pM
      */
      val (alpha, beta) = paramsTMat(n)
      (alpha, beta, gamma(n))
    })
  }
}
