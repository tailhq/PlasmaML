package io.github.mandar2812.PlasmaML.helios.core

import io.github.mandar2812.dynaml.evaluation.MetricsTF
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow._

/**
  *
  * */
class HeliosOmniTSMetrics[P: TF: IsFloatOrDouble](
  predictions: Tensor[P], targets: Tensor[P],
  size_causal_window: Int, time_scale: Tensor[P],
  p: Double = 2.0) extends
  MetricsTF[P](Seq("weighted_avg_err"), predictions, targets) {

  private[this] val scaling = Tensor(size_causal_window.toDouble-1d).castTo[P]

  override protected def run(): Tensor[P] = {

    val y = predictions(::, 0)

    val timelags = predictions(::, 1).sigmoid.multiply(scaling)

    val size_batch = predictions.shape(0)

    val repeated_times = dtf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val index_times = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(size_causal_window)).castTo[P]

    val repeated_index_times = dtf.stack(Seq.fill(size_batch)(index_times), axis = 0).castTo[P]

    val repeated_preds = dtf.stack(Seq.fill(size_causal_window)(y), axis = -1)

    val repeated_time_scales = dtf.stack(Seq.fill(size_causal_window)(time_scale), axis = -1)

    val convolution_kernel = (repeated_index_times - repeated_times)
      .abs
      .pow(Tensor(p).castTo[P])
      .multiply(Tensor(-1.0/p).castTo[P])
      .divide(repeated_time_scales)
      .exp

    val weighted_loss_tensor =
      (repeated_preds - targets)
        .square
        .multiply(convolution_kernel)
        .sum(axes = 1)
        .divide(convolution_kernel.sum(axes = 1))
        .mean[Int]()

    weighted_loss_tensor
  }

  override def print(): Unit = {
    println("\nModel Performance: "+name)
    println("============================")
    println()
    println(names.head+": "+results.scalar.asInstanceOf[Double])
    println()
  }
}
