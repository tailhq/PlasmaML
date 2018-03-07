package io.github.mandar2812.PlasmaML.dynamics.nn

import org.platanios.tensorflow.api.{Output, Shape, tf}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer}

//TODO: Transfer file to DynaML
/**
  * Represents a Continuous Time Recurrent Neural Network (CTRNN)
  * The layer simulates the discretized dynamics of the CTRNN for
  * a fixed number of time steps.
  *
  * @author mandar2812 date: 2018/03/06
  * */
case class FiniteHorizonCTRNN(
  override val name: String, units: Int,
  horizon: Int, timestep: Double,
  weightsInitializer: Initializer = RandomNormalInitializer(),
  biasInitializer: Initializer = RandomNormalInitializer(),
  gainInitializer: Initializer = RandomNormalInitializer(),
  timeConstantInitializer: Initializer = RandomNormalInitializer()) extends
  Layer[Output, Output](name){

  override val layerType: String = "FHCTRNN"

  override protected def _forward(input: Output, mode: Mode): Output = {

    val weights = tf.variable("Weights", input.dataType, Shape(units, units), weightsInitializer)
    val bias = tf.variable("Bias", input.dataType, Shape(units), biasInitializer)
    val timeconstant = tf.variable("TimeConstant", input.dataType, Shape(units), timeConstantInitializer)
    val gain = tf.variable("TimeConstant", input.dataType, Shape(units, units), timeConstantInitializer)

    tf.stack(
      (1 to horizon).scanLeft(input)((x, _) => {
        val decay = timeconstant.multiply(-1d).multiply(x)
        val interaction = x.tensorDot(gain, Seq(1), Seq(0)).add(bias).tanh.tensorDot(weights, Seq(1), Seq(0))

        x.add(decay).add(interaction)
      }).tail,
      axis = -1)

  }
}
