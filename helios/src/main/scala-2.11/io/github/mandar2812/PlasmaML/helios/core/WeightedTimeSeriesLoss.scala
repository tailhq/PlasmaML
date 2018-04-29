package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops.Output

case class WeightedTimeSeriesLoss(
  override val name: String,
  size_causal_window: Int,
  prior_wt: Double = 1.5,
  temperature: Double = 1.0,
  prior_type: String = "Hellinger") extends
  Loss[(Output, Output)](name) {

  override val layerType: String = s"WTSLoss[horizon:$size_causal_window]"

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    val preds   = input._1(::, 0::size_causal_window)
    val prob    = input._1(::, size_causal_window::).softmax()
    val targets = input._2

    val model_errors = preds.subtract(targets)

    val prior_prob = model_errors.square.multiply(-1.0).divide(temperature).softmax()

    def kl(prior: Output, p: Output): Output =
      prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

    val m = prior_prob.add(prob).divide(2.0)

    val prior_term =
      if(prior_type == "Jensen-Shannon") kl(prior_prob, m).add(kl(prob, m)).multiply(0.5)
      else if(prior_type == "Hellinger") prior_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))
      else if(prior_type == "Cross-Entropy") prior_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()
      else if(prior_type == "Kullback-Leibler") kl(prior_prob, prob)
      else Tensor(0.0).toOutput

    model_errors.square.multiply(prob.add(1.0)).sum(axes = 1).mean().add(prior_term.multiply(prior_wt))
  }
}

case class WeightedTimeSeriesLossSO(
  override val name: String,
  size_causal_window: Int,
  prior_wt: Double = 1.5,
  temperature: Double = 1.0,
  prior_type: String = "Jensen-Shannon") extends
  Loss[(Output, Output)](name) {

  override val layerType: String = s"WTSLossSO[horizon:$size_causal_window]"

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    val preds               = input._1(::, 0)
    val repeated_preds      = tf.stack(Seq.fill(size_causal_window)(preds), axis = -1)
    val prob                = input._1(::, 1::).softmax()
    val targets             = input._2

    val model_errors = repeated_preds.subtract(targets)

    val prior_prob = model_errors.square.multiply(-1.0).divide(temperature).softmax()

    def kl(prior: Output, p: Output): Output =
      prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

    val m = prior_prob.add(prob).divide(2.0)

    val prior_term =
      if(prior_type == "Jensen-Shannon") kl(prior_prob, m).add(kl(prob, m)).multiply(0.5)
      else if(prior_type == "Hellinger") prior_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))
      else if(prior_type == "Cross-Entropy") prior_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()
      else if(prior_type == "Kullback-Leibler") kl(prior_prob, prob)
      else Tensor(0.0).toOutput

    model_errors.square.multiply(prob.add(1.0)).sum(axes = 1).mean().add(prior_term.multiply(prior_wt))
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