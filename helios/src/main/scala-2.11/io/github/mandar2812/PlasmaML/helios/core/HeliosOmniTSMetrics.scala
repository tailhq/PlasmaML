package io.github.mandar2812.PlasmaML.helios.core

import io.github.mandar2812.dynaml.evaluation.MetricsTF
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow._

/**
  *
  * */
//TODO:: Check implementation
class HeliosOmniTSMetrics(
  preds: Tensor, targets: Tensor,
  size_causal_window: Int, time_scale: tf.Variable) extends
  MetricsTF(Seq("weighted_avg_err"), preds, targets) {

  private[this] val scaling = Tensor(size_causal_window.toDouble)

  override protected def run(): Tensor = {

    val y = preds(::, 0)

    val timelags = preds(::, 1).sigmoid.multiply(scaling).cast(INT32)

    val size_batch = preds.shape(0)

    val repeated_times = dtf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val index_times = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(1, size_causal_window))

    val repeated_index_times = dtf.stack(Seq.fill(size_batch)(index_times), axis = 0)
    val repeated_preds = dtf.stack(Seq.fill(size_causal_window)(y), axis = -1)

    val convolution_kernel = (repeated_index_times - repeated_times)
      .square
      .multiply(-0.5)
      .divide(time_scale.evaluate())
      .exp

    val weighted_loss_tensor =
      (repeated_preds - targets)
        .multiply(convolution_kernel)
        .sum(axes = 1)
        .divide(
          convolution_kernel.sum(axes = 1, keepDims = true)
        )

    weighted_loss_tensor
  }
}
