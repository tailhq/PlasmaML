package io.github.mandar2812.PlasmaML.dynamics.diffusion

import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, MAKernel}
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.pipes.Encoder
import io.github.mandar2812.dynaml.DynaMLPipe._

/**
  * Extension of [[io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior]], adapted
  * for the plasma diffusion scheme.
  *
  * @author mandar2812 date 27/06/2017.
  * */
class DiffusionPrior(
  val trend: MagnetosphericProcessTrend[Map[String, Double]],
  val covarianceSpace: LocalScalarKernel[Double],
  val covarianceTime: LocalScalarKernel[Double],
  val noise: Double = 0.5,
  initialParams: (Double, Double, Double, Double, Double) = (1.0, 10d, 0d, 0.506, -9.325)) extends
  CoRegGPPrior[Double, Double, Map[String, Double]](
    covarianceSpace, covarianceTime,
    new MAKernel(noise), new MAKernel(noise),
    Encoder(identityPipe[Map[String, Double]], identityPipe[Map[String, Double]])) {


  protected var parameters: (Double, Double, Double, Double, Double) = initialParams

  override def _meanFuncParams = trend.transform.i(parameters)

  override def meanFuncParams_(p: Map[String, Double]) = {
    parameters = trend.transform(p)
  }

  override val meanFunctionPipe = trend
}
