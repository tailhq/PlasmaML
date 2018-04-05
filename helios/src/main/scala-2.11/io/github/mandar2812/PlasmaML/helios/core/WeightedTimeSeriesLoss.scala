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

    //val log_temperature: tf.Variable = tf.variable("time_scale", FLOAT32, Shape(), tf.OnesInitializer)

    val preds   = input._1(::, 0::size_causal_window)
    val prob    = input._1(::, size_causal_window::)/*.divide(log_temperature.exp)*/.softmax()
    val targets = input._2

    val prior_prob_time_lags = preds.subtract(targets).square.softmax()

    val kl_divergence = prior_prob_time_lags.divide(prob).log.multiply(prior_prob_time_lags).sum(axes = 1).mean()

    val entropy = prob.log.multiply(prob.multiply(-1.0)).sum(axes = 1).mean()
    //val prob = unorm_prob.divide(tf.stack(Seq.fill(size_causal_window)(unorm_prob.sum(axes = 1)), axis = -1))

    preds.subtract(input._2).square.multiply(prob.add(1.0)).sum(axes = 1).mean().add(kl_divergence).add(entropy)
  }
}


case class MOGrangerLoss(
  override val name: String,
  size_causal_window: Int,
  error_exponent: Double = 2.0,
  weight_error: Double,
  scale_lags: Boolean = true) extends
  Loss[(Output, Output)](name) {

  override val layerType = s"MOGrangerLoss[horizon:$size_causal_window]"

  private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

  override protected def _forward(input: (Output, Output), mode: Mode) = {

    val alpha = Tensor(1.0)
    val nu = Tensor(1.0)
    val q = Tensor(1.0)


    val predictions        = input._1(::, 0 :: -1)
    val unscaled_lags      = input._1(::, -1)

    val targets            = input._2


    val timelags           = if (scale_lags) {
      unscaled_lags
        .multiply(alpha.add(1E-6).square.multiply(-1.0))
        .exp
        .multiply(q.square)
        .add(1.0)
        .pow(nu.square.pow(-1.0).multiply(-1.0))
        .multiply(scaling)
    } else {
      unscaled_lags
    }

    val repeated_times      = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val index_times: Output = Tensor(
      (0 until size_causal_window).map(_.toDouble)
    ).reshape(
      Shape(size_causal_window)
    )

    val error_tensor = predictions.subtract(targets)

    val convolution_kernel_temporal = error_tensor
      .abs.pow(error_exponent)
      .l2Normalize(axes = 1)
      .square
      .subtract(1.0)
      .multiply(-1/2.0)

    val weighted_temporal_loss_tensor = repeated_times
      .subtract(index_times)
      .square
      .multiply(convolution_kernel_temporal)
      .sum(axes = 1)
      .divide(convolution_kernel_temporal.sum(axes = 1))
      .mean()

    error_tensor.square.sum(axes = 1).multiply(0.5*weight_error).mean().add(weighted_temporal_loss_tensor)
  }
}