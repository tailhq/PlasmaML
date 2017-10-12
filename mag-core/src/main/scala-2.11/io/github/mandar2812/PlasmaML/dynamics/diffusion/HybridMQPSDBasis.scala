package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.RadialBasis
import io.github.mandar2812.dynaml.pipes.{Basis, DataPipe}
import io.github.mandar2812.dynaml.analysis.implicits._

/**
  * <h3>Phase Space Density: Hybrid MQ &amp; Inverse MQ Mesh-Based Radial Basis<h3>
  *
  * Implements a multi-quadric radial basis expansion for the spatial component
  * and a inverse multi-quadric radial basis expansion for the temporal component of the
  * phase space density with nodes placed on a regular space time mesh.
  *
  * */
class HybridMQPSDBasis(beta_t: Double)(
  lShellLimits: (Double, Double), nL: Int,
  timeLimits: (Double, Double), nT: Int,
  logScales: (Boolean, Boolean) = (false, false))
  extends PSDRadialBasis(
    lShellLimits, nL,
    timeLimits, nT,
    logScales) {

  require(beta_t > 0, "Beta parameter must be positive")

  val (spatial_activation, temporal_activation) = (
    RadialBasis.multiquadric,
    RadialBasis.invMultiQuadric(beta_t))

  val (spatial_field, temporal_field) = (innerProdDouble, innerProdDouble)

  /**
    * Calculate the function which must be multiplied to the current
    * basis in order to obtain the operator transformed basis.
    **/
  override def operator_basis(
    diffusionField: DataPipe[(Double, Double), Double],
    diffusionFieldGradL: DataPipe[(Double, Double), Double],
    lossTimeScale: DataPipe[(Double, Double), Double]): Basis[(Double, Double)] =
    Basis((x: (Double, Double)) => {

      val beta_l = -1d
      val (l, t) = x
      val dll = diffusionField(x)
      val alpha = diffusionFieldGradL(x) - 2d*diffusionField(x)/x._1
      val lambda = lossTimeScale(x)

      DenseVector(
        centers.zip(scales).map(c => {
          val ((lc, tc), (theta_s, theta_t)) = c

          val (d_l, d_t) = (spatial_field.minus(l, lc), temporal_field.minus(t, tc))

          val scaledDist = (d_l/theta_s, d_t/theta_t)

          val (f_l, f_t) = (
            1d + spatial_field.dot(scaledDist._1, scaledDist._1),
            1d + temporal_field.dot(scaledDist._2, scaledDist._2))

          val sq = (s: Double) => s*s

          val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

          beta_l*invThetaS*dll*(sq(d_l)*(0.5*beta_l+1) - f_l)/math.pow(f_l, 2+0.5*beta_l) -
            beta_l*invThetaS*alpha*math.abs(d_l)/f_l +
            beta_t*invThetaT*math.abs(d_t)/f_t -
            lambda
        }).toArray)
    })

  override protected val f: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => DenseVector(
      centers.zip(scales).map(cs => {
        val (l, t) = x
        val ((lc, tc), (theta_s, theta_t)) = cs

        val (d_l, d_t) = (spatial_field.minus(l, lc), temporal_field.minus(t, tc))

        val scaledDist = (d_l/theta_s, d_t/theta_t)


        val (r_s, r_t) = (
          math.sqrt(spatial_field.dot(scaledDist._1, scaledDist._1)),
          math.sqrt(temporal_field.dot(scaledDist._2, scaledDist._2))
          )

        spatial_activation(r_s)*temporal_activation(r_t)
      }).toArray
    )
}
