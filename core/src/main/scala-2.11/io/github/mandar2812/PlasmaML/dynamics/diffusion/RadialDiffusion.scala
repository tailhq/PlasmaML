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
  * df/dt = L<sup>2</sup>d/dL[D<sub>LL</sub> &times L<sup>-2</sup> &times;  df/dL] + Q(L,t)
  *
  * This class solves the radial diffusion dynamics by representing the forward model
  * as a feed forward neural network represented by DynaML's [[NeuralStack]].
  *
  * @param lShellLimits The minimum and maximum value of L* i.e. the drift shell
  * @param timeLimits The minimum and maximum of the time coordinate.
  * @param nL The number of bins to divide spatial domain into.
  * @param nT The number of bins to divide temporal domain into.
  * @param linearDecay Set to true if injection term Q(L, t) = - &lambda;(L,t) &times; f(L,t),
  *                    in this case the [[solve()]] method will accept the realisation of &lambda;(L,t)
  *                    instead of Q(L,t).
  *
  * */
class RadialDiffusion(
  lShellLimits: (Double, Double),
  timeLimits: (Double, Double),
  nL: Int, nT: Int,
  linearDecay: Boolean = true) {

  val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

  val stackFactory = NeuralStackFactory(
      (1 to nT).map(_ => new NeuralLayerFactory(RadialDiffusionLayer.forwardPropagate, VectorLinear)):_*
  )

  val computeStackParameters =
    if(linearDecay) DataPipe3(RadialDiffusion.getLinearDecayModelParams(lShellLimits, timeLimits, nL, nT))
    else DataPipe3(RadialDiffusion.getInjectionModelParams(lShellLimits, timeLimits, nL, nT))

  def getComputationStack = computeStackParameters > stackFactory

  /**
    * Solve the radial diffusion dynamics.
    *
    * @param injectionProfile Depending on value the [[linearDecay]] flag,
    *                         it is the injection or linear decay profile
    * @param diffusionProfile The radial diffusion coefficient as a field
    *                         realised on a uniform grid
    * @param boundaryFlux The phase space density at the boundaries of the
    *                     spatial domain (drift shell) for each time epoch
    * @param f0 The initial phase space density profile realised on the spatial grid
    *
    * @return The phase space density evolution as a [[Seq]] of [[DenseVector]]
    *
    * */
  def solve(
    injectionProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double])(
    f0: DenseVector[Double]): Seq[DenseVector[Double]] =
    getComputationStack(injectionProfile, diffusionProfile, boundaryFlux) forwardPropagate f0

}

object RadialDiffusion {

  /**
    * The forward difference convolution filter to
    * be applied on the adjusted diffusion field.
    * */
  def forwardConvDiff(center_i: Int, center_j: Int)(i: Int, j: Int): Double =
    if(i - center_i >= 0 && i - center_i <= 1 && j - center_j >= 0 && j - center_j <= 1) 0.25 else 0.0

  /**
    * The backward difference convolution filter to be
    * applied on the adjusted diffusion field.
    * */
  def backwardConvDiff(center_i: Int, center_j: Int)(i: Int, j: Int): Double =
    if(center_i - i >= 0 && center_i - i <= 1 && j - center_j >= 0 && j - center_j <= 1) 0.25 else 0.0

  /**
    * Forward difference convolution filter for boundary flux
    * */
  def forwardConvBoundaryFlux(center_i:Int, center_j: Int)(i: Int, j: Int): Double =
    if(j == center_j && i == center_i) -1.0 else if(j == center_j+1 && i == center_i) 1.0 else 0.0

  /**
    * Forward difference convolution filter for loss profiles/scales
    * */
  def forwardConvLossProfile(center_i: Int, center_j: Int)(i: Int, j: Int): Double =
    if(i == center_i && j - center_j >= 0 && j - center_j <= 1) 0.5 else 0.0

  /**
    * Performs convolution of a [[DenseMatrix]] with a supplied filter.
    * */
  def conv(filter: (Int, Int) => Double)(data: DenseMatrix[Double]) =
    sum(data.mapPairs((coords, value) => value * filter(coords._1, coords._2)))

  /**
    * Compute [[NeuralStack]] parameters for forward model governed by
    * linear decay term.
    *
    * @param lShellLimits The lower and upper limits of L-Shell
    * @param timeLimits The lower and upper limits of time.
    * @param nL The number of partitions to create in the L-Shell domain.
    * @param nT The number of partitions to create in the time domain.
    * @return A sequence of parameters which specify the [[NeuralStack]]
    *         used to compute the radial diffusion solution.
    * */
  def getLinearDecayModelParams(
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

  /**
    * Compute [[NeuralStack]] parameters for a general injection
    * diffusion model.
    *
    * @param lShellLimits The lower and upper limits of L-Shell
    * @param timeLimits The lower and upper limits of time.
    * @param nL The number of partitions to create in the L-Shell domain.
    * @param nT The number of partitions to create in the time domain.
    * @return A sequence of parameters which specify the [[NeuralStack]]
    *         used to compute the radial diffusion solution.
    * */
  def getInjectionModelParams(
    lShellLimits: (Double, Double), timeLimits: (Double, Double), nL: Int, nT: Int)(
    injectionProfile: DenseMatrix[Double], diffusionProfile: DenseMatrix[Double], boundaryFlux: DenseMatrix[Double])
  : Seq[(Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double])] = {

    val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

    val lSqVec = square(DenseVector.tabulate[Double](nL + 1)(i =>
      if(i < nL) lShellLimits._1+(deltaL*i) else lShellLimits._2))

    val adjLVec = lSqVec.map(v => 0.5*v/(deltaL*deltaL))

    val adjustedDiffusionProfile = diffusionProfile.mapPairs((coords, value) => value/lSqVec(coords._1))

    val invDeltaT = 1/deltaT

    val paramsTMat: (Int) => (Seq[Seq[Double]], Seq[Seq[Double]]) = (n) => {

      val (alph, bet) = (1 until nL).map(j => {
        val b = adjLVec(j)*(
          conv(forwardConvDiff(j, n))(adjustedDiffusionProfile) +
          conv(backwardConvDiff(j, n))(adjustedDiffusionProfile))

        val a = -adjLVec(j)*conv(backwardConvDiff(j, n))(adjustedDiffusionProfile)

        val c = -adjLVec(j)*conv(forwardConvDiff(j, n))(adjustedDiffusionProfile)

        (Seq(-a*deltaT, 1d - b*deltaT, -c*deltaT), Seq(a*deltaT, 1d + b*deltaT, c*deltaT))
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

    val delta: (Int) => DenseVector[Double] = (n) => {
      DenseVector.tabulate[Double](nL + 1)(j => deltaT*conv(forwardConvLossProfile(j,n))(injectionProfile))
    }

    /*
    * Instantiate layer transformations
    * */
    (0 until nT).map(n => {
      val (alpha, beta) = paramsTMat(n)
      (alpha, beta, gamma(n) + delta(n))
    })
  }
}
