package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import io.github.mandar2812.dynaml.algebra.square
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe3}

/**
  *
  * Implementation of a discrete radial diffusion system.
  *
  * df/dt = L<sup>2</sup>d/dL(D<sub>LL</sub> &times; L<sup>-2</sup> &times;  df/dL) + Q(L,t)
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
  * @author mandar2812 date 30/03/2017.
  *
  * */
class RadialDiffusion(
  lShellLimits: (Double, Double),
  timeLimits: (Double, Double),
  nL: Int, nT: Int,
  linearDecay: Boolean = true) {

  val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)


  val lShellVec = DenseVector.tabulate[Double](nL+1)(i =>
    if(i < nL) lShellLimits._1+(deltaL*i)
    else lShellLimits._2)

  val timeVec = DenseVector.tabulate[Double](nT+1)(i =>
    if(i < nT) timeLimits._1+(deltaT*i)
    else timeLimits._2)

  val stackFactory = DataPipe(
    (params: Stream[(Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double])]) =>
      new LazyNeuralStack[(Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double]), DenseVector[Double]](
        params.map((layer) => RadialDiffusionLayer(layer._1, layer._2, layer._3)))
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
    * @return The phase space density evolution as a [[Stream]] of [[DenseVector]]
    *
    * */
  def solve(
    injectionProfile: DenseMatrix[Double],
    diffusionProfile: DenseMatrix[Double],
    boundaryFlux: DenseMatrix[Double])(
    f0: DenseVector[Double]): Stream[DenseVector[Double]] =
    getComputationStack(injectionProfile, diffusionProfile, boundaryFlux) forwardPropagate f0

  /**
    * Solve the radial diffusion dynamics.
    *
    * @param injection Depending on value the [[linearDecay]] flag,
    *                         it is the injection or linear decay field
    * @param diffusionField The radial diffusion coefficient as a field
    * @param boundaryFlux The phase space density at the boundaries of the
    *                     spatial domain (drift shell) as a function of LShell and time.
    * @param f0 The initial phase space density profile as a function of LShell
    *
    * @return The phase space density evolution as a [[Stream]] of [[DenseVector]]
    *
    * */
  def solve(
    injection: (Double, Double) => Double,
    diffusionField: (Double, Double) => Double,
    boundaryFlux: (Double, Double) => Double)(
    f0: (Double) => Double): Stream[DenseVector[Double]] = {

    val initialPSD: DenseVector[Double] = DenseVector(lShellVec.map(l => f0(l)).toArray)

    val diffProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => diffusionField(lShellVec(i), timeVec(j)))

    val injectionProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => injection(lShellVec(i), timeVec(j)))

    val boundFlux = DenseMatrix.tabulate[Double](nL+1,nT)((i,j) => {
      if(i == nL || i == 0) boundaryFlux(lShellVec(i), timeVec(j))
      else 0.0
    })

    solve(injectionProfile, diffProfile, boundFlux)(initialPSD)
  }

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

  def apply(
    lShellLimits: (Double, Double), timeLimits: (Double, Double),
    nL: Int, nT: Int, linearDecay: Boolean = true) =
    new RadialDiffusion(lShellLimits, timeLimits, nL, nT, linearDecay)

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
    boundaryFlux: DenseMatrix[Double]): Stream[(Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double])] = {

    val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

    val lSqVec = square(DenseVector.tabulate[Double](nL + 1)(i =>
      if(i < nL) lShellLimits._1+(deltaL*i) else lShellLimits._2))

    val adjLVec = lSqVec.map(v => 0.5*v/(deltaL*deltaL))

    val adjustedDiffusionProfile = diffusionProfile.mapPairs((coords, value) => value/lSqVec(coords._1))

    val invDeltaT = 1/deltaT

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
    Stream.tabulate(nT)(n => {

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
  : Stream[(Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double])] = {

    val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)

    val lSqVec = square(DenseVector.tabulate[Double](nL + 1)(i =>
      if(i < nL) lShellLimits._1+(deltaL*i) else lShellLimits._2))

    val adjLVec = lSqVec.map(v => 0.5*v/(deltaL*deltaL))

    val adjustedDiffusionProfile = diffusionProfile.mapPairs((coords, value) => value/lSqVec(coords._1))

    val invDeltaT = 1/deltaT

    val paramsTMat: (Int) => (Seq[Seq[Double]], Seq[Seq[Double]]) = (n) => {

      val (alpha_n, beta_n) = (1 until nL).map(j => {
        val b = adjLVec(j)*(
          conv(forwardConvDiff(j, n))(adjustedDiffusionProfile) +
          conv(backwardConvDiff(j, n))(adjustedDiffusionProfile))

        val a = -adjLVec(j)*conv(backwardConvDiff(j, n))(adjustedDiffusionProfile)

        val c = -adjLVec(j)*conv(forwardConvDiff(j, n))(adjustedDiffusionProfile)

        (Seq(-a, invDeltaT - b, -c), Seq(a, invDeltaT + b, c))
      }).unzip

      (
        Seq(Seq(0.0, 1.0, 0.0)) ++ alpha_n ++ Seq(Seq(0.0, 1.0, 0.0)),
        Seq(Seq(0.0, 1.0, 0.0)) ++ beta_n ++ Seq(Seq(0.0, 1.0, 0.0)))
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
      DenseVector.tabulate[Double](nL + 1)(j => conv(forwardConvLossProfile(j,n))(injectionProfile))
    }

    /*
    * Instantiate layer transformations
    * */
    Stream.tabulate(nT)(n => {
      val (alpha, beta) = paramsTMat(n)
      (alpha, beta, gamma(n) + delta(n))
    })
  }
}
