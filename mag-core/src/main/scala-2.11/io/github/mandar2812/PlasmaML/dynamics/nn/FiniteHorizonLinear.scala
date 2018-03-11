package io.github.mandar2812.PlasmaML.dynamics.nn

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer}
import org.platanios.tensorflow.api.{---, Output, Shape, tf}

/**
  * Projection of a finite horizon multivariate
  * time series onto an observation space.
  *
  * @param units The degrees of freedom or dimensionality of the dynamical system
  * @param observables The dimensionality of the observations at each time epoch.
  * @author mandar2812 date 11/03/2018
  * */
//TODO: Remove after it is integrated into DynaML
case class FiniteHorizonLinear(
  override val name: String,
  units: Int, observables: Int, horizon: Int,
  weightsInitializer: Initializer = RandomNormalInitializer(),
  biasInitializer: Initializer = RandomNormalInitializer()) extends
  Layer[Output, Output](name) {

  override val layerType: String = "FHLinear"

  override protected def _forward(input: Output, mode: Mode): Output = {
    val weights      = tf.variable("Weights", input.dataType, Shape(observables, units), weightsInitializer)
    val bias         = tf.variable("Bias", input.dataType, Shape(observables), biasInitializer)

    tf.stack(
      (0 until horizon).map(i => {
        input(---, i).tensorDot(weights, Seq(1), Seq(1)).add(bias)
      }),
      axis = -1)
  }
}
