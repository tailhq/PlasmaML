package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.RadialBasis
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.pipes._

/**
  * <h3>Phase Space Density: Gaussian Mesh-Based Radial Basis<h3>
  *
  * Implements a gaussian radial basis expansion for the phase space density
  * with nodes placed on a regular space time mesh.
  *
  * */
class GaussianPSDBasis(
  lShellLimits: (Double, Double), nL: Int,
  timeLimits: (Double, Double), nT: Int,
  logScales: (Boolean, Boolean) = (false, false))
  extends PSDRadialBasis(
    lShellLimits, nL,
    timeLimits, nT,
    logScales) {

  val (basisL, basisT) = (
    RadialBasis.gaussianBasis(lSeq, scalesL, bias = false),
    RadialBasis.gaussianBasis(tSeq, scalesT, bias = false))

  val productBasis: Basis[(Double, Double)] = basisL*basisT

  override protected val f: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => productBasis(x)

  /**
    * Calculate the function which must be multiplied to the current
    * basis in order to obtain the operator transformed basis.
    **/
  override def operator_basis(
    diffusionField: DataPipe[(Double, Double), Double],
    diffusionFieldGradL: DataPipe[(Double, Double), Double],
    lossTimeScale: DataPipe[(Double, Double), Double]): Basis[(Double, Double)] =
    Basis((x: (Double, Double)) => {

      val (l, t) = x

      val dll = diffusionField(x)
      val alpha = diffusionFieldGradL(x) - 2d*diffusionField(x)/x._1
      val lambda = lossTimeScale(x)

      DenseVector(
        centers.zip(scales).map(c => {
          val ((l_tilda, t_tilda), (theta_s, theta_t)) = c

          val grT = (i: Int, j: Int) => gradSqNormDouble(i, j)(t, t_tilda)
          val grL = (i: Int, j: Int) => gradSqNormDouble(i, j)(l, l_tilda)


          val sq = (s: Double) => s*s

          val (invThetaS, invThetaT) = (sq(1/theta_s), sq(1/theta_t))

          val gs = 0.5*invThetaS*sq(grL(1, 0)) - grL(2, 0)

          0.5*invThetaS*dll*gs - 0.5*invThetaS*alpha*grL(1, 0) + 0.5*invThetaT*grT(1, 0) - lambda
        }).toArray) *:*
        f(x)
    })

}
