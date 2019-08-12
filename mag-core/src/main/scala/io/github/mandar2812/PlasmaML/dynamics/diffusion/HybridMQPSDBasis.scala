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
class HybridMQPSDBasis(
  beta_t: Double
)(lShellLimits: (Double, Double),
  nL: Int,
  timeLimits: (Double, Double),
  nT: Int,
  logScales: (Boolean, Boolean) = (false, false))
    extends PSDRadialBasis(lShellLimits, nL, timeLimits, nT, logScales) {

  require(beta_t > 0, "Beta parameter must be positive")

  val (spatial_activation, temporal_activation) =
    (RadialBasis.multiquadric, RadialBasis.invMultiQuadric(beta_t))

  val (spatial_field, temporal_field) = (innerProdDouble, innerProdDouble)

  override protected val f: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) =>
      DenseVector(
        centers
          .zip(scales)
          .map(cs => {
            val (l, t)                         = x
            val ((lc, tc), (theta_s, theta_t)) = cs

            val (d_l, d_t) =
              (spatial_field.minus(l, lc), temporal_field.minus(t, tc))

            val scaledDist = (d_l / theta_s, d_t / theta_t)

            val (r_s, r_t) = (
              math.sqrt(spatial_field.dot(scaledDist._1, scaledDist._1)),
              math.sqrt(temporal_field.dot(scaledDist._2, scaledDist._2))
            )

            spatial_activation(r_s) * temporal_activation(r_t)
          })
          .toArray
      )

  override val f_l: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => {
      val beta_l = -1d
      val (l, _) = x

      DenseVector(
        centers
          .zip(scales)
          .map(cs => {
            val ((lc, _), (theta_s, _)) = cs

            val d_l = spatial_field.minus(l, lc)

            val scaledDist = d_l / theta_s

            val g_l = 1d + scaledDist * scaledDist

            val invThetaS = 1 / theta_s

            -beta_l * invThetaS * math.abs(d_l) / g_l
          })
          .toArray
      ) *:*
        f(x)
    }

  override val f_ll: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => {
      val beta_l = -1d
      val (l, _) = x

      DenseVector(
        centers
          .zip(scales)
          .map(cs => {
            val ((lc, _), (theta_s, _)) = cs

            val d_l = spatial_field.minus(l, lc)

            val scaledDist = d_l / theta_s

            val g_l = 1d + scaledDist * scaledDist

            val invThetaS = 1 / theta_s

            val sq = (s: Double) => s * s

            beta_l * invThetaS * (sq(d_l) * (0.5 * beta_l + 1) - g_l) / math
              .pow(g_l, 2 + 0.5 * beta_l)
          })
          .toArray
      ) *:*
        f(x)
    }

  override val f_t: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => {

      val (_, t) = x

      DenseVector(
        centers
          .zip(scales)
          .map(cs => {
            val ((_, tc), (_, theta_t)) = cs

            val d_t = temporal_field.minus(t, tc)

            val scaledDist = d_t / theta_t

            val g_t = 1d + scaledDist * scaledDist

            val invThetaT = 1 / theta_t

            -beta_t * invThetaT * math.abs(d_t) / g_t
          })
          .toArray
      ) *:*
        f(x)
    }

}
