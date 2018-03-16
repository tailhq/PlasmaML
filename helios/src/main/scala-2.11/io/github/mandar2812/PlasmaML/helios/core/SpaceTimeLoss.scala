package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.{::, Shape, Tensor, tf}

case class SpaceTimeLoss(
  override val name: String,
  size_causal_window: Int) extends
  Loss[(Output, Output)](name) {

  override val layerType: String = s"SpaceTimeLoss[horizon:$size_causal_window]"

  private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    //Obtain section corresponding to velocity predictions
    val predictions = input._1(::, 0)

    val targets = input._2

    val timelags = input._1(::, 1).sigmoid.multiply(scaling)

    val repeated_times = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(predictions), axis = -1)

    val index_times: Output = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(size_causal_window))


    val time_loss = repeated_times.subtract(index_times)
      .square.multiply(0.5)

    val space_loss = repeated_preds.subtract(targets).square.multiply(0.5)

    space_loss.add(time_loss).sum(axes = 1).mean()
  }
}
