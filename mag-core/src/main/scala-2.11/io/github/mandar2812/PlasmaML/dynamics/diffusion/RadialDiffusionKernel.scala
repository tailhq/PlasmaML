package io.github.mandar2812.PlasmaML.dynamics.diffusion

import io.github.mandar2812.dynaml.kernels.{GenExpSpaceTimeKernel, LinearPDEKernel, LocalScalarKernel}
import io.github.mandar2812.dynaml.pipes.{DataPipe2, MetaPipe}

/**
  * Radial Diffusion Gaussian Process
  *
  * @author mandar2812 date 07/07/2017.
  * */
abstract class RadialDiffusionKernel[I](
  sigma: Double, theta_space: Double,
  theta_time: Double)(
  val ds: DataPipe2[I, I, Double],
  val dt: DataPipe2[Double, Double, Double]) extends
  LinearPDEKernel[I] {


  override val baseKernel: GenExpSpaceTimeKernel[I] =
    new GenExpSpaceTimeKernel[I](sigma, theta_space, theta_time)(ds, dt)

  val diffusionField: MetaPipe[Map[String, Double], (I, Double), Double]

  val lossTimeScale: MetaPipe[Map[String, Double], (I, Double), Double]


}
