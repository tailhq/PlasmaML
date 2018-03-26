package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops.Output

case class WeightedTimeSeriesLoss(
  override val name: String,
  size_causal_window: Int) extends
  Loss[(Output, Output)](name) {

  override val layerType: String = s"WTSLoss[horizon:$size_causal_window]"

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    val log_temperature: tf.Variable = tf.variable("time_scale", FLOAT32, Shape(), tf.OnesInitializer)

    val preds = input._1(::, 0::size_causal_window)
    val unorm_prob = input._1(::, size_causal_window::).divide(log_temperature.exp).exp

    val prob = unorm_prob.divide(tf.stack(Seq.fill(size_causal_window)(unorm_prob.sum(axes = 1)), axis = -1))

    preds.subtract(input._2).square.multiply(prob).sum(axes = 1).mean()
  }
}
