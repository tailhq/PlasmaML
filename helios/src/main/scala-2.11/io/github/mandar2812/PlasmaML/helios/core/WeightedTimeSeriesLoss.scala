package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>Dynamic time-lag Inference: Weighted Loss.</h3>
  *
  * The loss functions assumes the prediction architecture
  * returns 2 &times; h number of outputs, where h = [[size_causal_window]].
  *
  * The first h outputs are predictions of the target signal, for
  * each time step in the horizon.
  *
  * The next h outputs are unscaled probability predictions regarding
  * the causal time lag between the input and output signals, computed
  * for any input pattern..
  *
  * The loss term for each data sample in the mini-batch is computed
  * independently as follows.
  *
  * L = &Sigma; <sub>i</sub> (|f<sub>i</sub> - y<sub>i</sub>|<sup>2</sup> &times; (1 + p<sub>i</sub>)
  * + &gamma; (&sqrt; p<sup>*</sup><sub>i</sub> - &sqrt; p<sub>i</sub>)<sup>2</sup>)
  *
  * @param name A string identifier for the loss instance.
  * @param size_causal_window The size of the finite time horizon
  *                           to look for causal effects.
  * @param prior_wt The multiplicative constant applied on the
  *                 "prior" term; i.e. a divergence term between the
  *                 predicted probability distribution of the
  *                 time lags and the so called
  *                 "target probability distribution".
  * @param temperature The Gibbs temperature which scales the softmax probability
  *                    transformation while computing the target probability distribution.
  *
  * @param prior_type The kind of divergence term to be used as a prior over the probability
  *                   distribution predicted for the time lag. Available options include.
  *                   <ul>
  *                     <li>Hellinger distance (default)</li>
  *                     <li>Kullback-Leibler Divergence</li>
  *                     <li>Cross Entropy</li>
  *                     <li>Jensen-Shannon Divergence</li>
  *                   </ul>
  *
  * */
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

    val target_prob = model_errors.square.multiply(-1.0).divide(temperature).softmax()

    def kl(prior: Output, p: Output): Output =
      prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

    val m = target_prob.add(prob).divide(2.0)

    val prior_term =
      if(prior_type == "Jensen-Shannon") kl(target_prob, m).add(kl(prob, m)).multiply(0.5)
      else if(prior_type == "Hellinger") target_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))
      else if(prior_type == "Cross-Entropy") target_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()
      else if(prior_type == "Kullback-Leibler") kl(target_prob, prob)
      else Tensor(0.0).toOutput

    model_errors.square.multiply(prob.add(1.0)).sum(axes = 1).mean().add(prior_term.multiply(prior_wt))
  }
}

case class WeightedTimeSeriesLossSO(
  override val name: String,
  size_causal_window: Int,
  prior_wt: Double = 1.5,
  temperature: Double = 1.0,
  prior_type: String = "Hellinger") extends
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

case class WeightedTimeSeriesLossBeta(
  override val name: String,
  size_causal_window: Int,
  batchSize: Int,
  prior_wt: Double = 1.5,
  temperature: Double = 1.0,
  prior_type: String = "Hellinger") extends
  Loss[(Output, Output)](name) {

  override val layerType: String = s"WTSLossBeta[horizon:$size_causal_window]"

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    //Extract the multi-variate predictions
    val preds         = input._1(::, 0::size_causal_window)

    //Extract parameters of the timalag predictive distribution
    val alpha_beta    = input._1(::, size_causal_window::).square.add(1.0)
    val (alpha, beta) = (alpha_beta(::, 0), alpha_beta(::, 1))

    val (stacked_alpha, stacked_beta) = (
      tf.stack(Seq.fill(size_causal_window)(alpha), axis = 1),
      tf.stack(Seq.fill(size_causal_window)(beta),  axis = 1))

    val index_times   = Tensor(0 until size_causal_window).reshape(Shape(1, size_causal_window)).toOutput

    val t             = tf.stack(Seq.fill(batchSize)(index_times), axis = 0)

    val n             = Tensor(size_causal_window - 1.0).toOutput
    val n_minus_t     = t.multiply(-1.0).add(n)

    val norm_const    = stacked_alpha.logGamma
      .add(stacked_beta.logGamma)
      .subtract(stacked_alpha.add(stacked_beta).logGamma)



    val prob          = t
      .add(stacked_alpha).logGamma
      .add(n_minus_t.add(stacked_beta).logGamma)
      .add(n.add(1.0).logGamma)
      .subtract(t.add(1.0).logGamma)
      .subtract(n_minus_t.add(1.0).logGamma)
      .subtract(norm_const).exp


    val targets       = input._2
    val model_errors  = preds.subtract(targets)
    val target_prob   = model_errors.square.multiply(-1.0).divide(temperature).softmax()

    def kl(prior: Output, p: Output): Output =
      prior.divide(p).log.multiply(prior).sum(axes = 1).mean()

    val m = target_prob.add(prob).divide(2.0)

    val prior_term =
      if(prior_type == "Jensen-Shannon") kl(target_prob, m).add(kl(prob, m)).multiply(0.5)
      else if(prior_type == "Hellinger") target_prob.sqrt.subtract(prob.sqrt).square.sum().sqrt.divide(math.sqrt(2.0))
      else if(prior_type == "Cross-Entropy") target_prob.multiply(prob.log).sum(axes = 1).multiply(-1.0).mean()
      else if(prior_type == "Kullback-Leibler") kl(target_prob, prob)
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