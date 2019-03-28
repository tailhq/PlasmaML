package io.github.mandar2812.PlasmaML.helios.core

import io.github.mandar2812.dynaml.utils.annotation.Experimental
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Output

@Experimental
/**
  * <h3>RBF-Kernel Weighted Solar Wind Loss (KSW Loss)</h3>
  *
  * A weighted loss function which enables fuzzy learning of
  * the solar wind propagation from heliospheric
  * images to ACE.
  *
  * @author mandar2812*
  * */
case class GenRBFSWLoss[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric: IsNotQuantized,
L: TF : IsFloatOrDouble](
  override val name:  String,
  size_causal_window: Int,
  corr_cutoff:        Double = -0.75,
  prior_weight:       Double = 1.0,
  prior_scaling:      Double = 1.0,
  batch:              Int    = -1)
  extends Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType: String = s"RBFSW[$size_causal_window]"

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    //Declare learnable parameters.
    val time_scale: tf.Variable[P] = tf.variable[P]("time_scale", Shape(), tf.OnesInitializer)
    val logp: tf.Variable[P]       = tf.variable[P]("logp", Shape(), tf.RandomUniformInitializer(0.0f, 1.0f))
    val p                          = logp.exp

    //Declare relevant quantities
    val predictions   = input._1._1
    val unscaled_lags = input._1._2
    val targets       = input._2
    val timelags      = unscaled_lags//.sigmoid.multiply(scaling)

    //Determine the batch size, if not provided @TODO: Iron out bugs in this segment
    val batchSize =
      if(batch == -1) predictions.size.toInt/predictions.shape.toTensor.prod().scalar
      else batch

    //Sort the predicted targets and time lags, obtaining
    //the ranks of each element.
    val rank_preds     = predictions.topK(batchSize)._2.castTo[P]
    val rank_unsc_lags = unscaled_lags.topK(batchSize)._2.castTo[P]

    //Standardize (mean center) the ranked tensors
    val (ranked_preds_mean, ranked_unscaled_lags_mean) = (rank_preds.mean(), rank_unsc_lags.mean())

    val (ranked_preds_std, ranked_unscaled_lags_std)   = (
      rank_preds.subtract(ranked_preds_mean).square.mean().sqrt,
      rank_unsc_lags.subtract(ranked_unscaled_lags_mean).square.mean().sqrt)

    val (norm_ranked_targets, norm_ranked_unsc_lags)   = (
      rank_preds.subtract(ranked_preds_mean).divide(ranked_preds_std),
      rank_unsc_lags.subtract(ranked_unscaled_lags_mean).divide(ranked_unscaled_lags_std))


    val repeated_times = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(predictions), axis = -1)

    val index_times: Output[P] =
      Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(size_causal_window)).castTo[P].toOutput

    //Calculate the convolution kernel of the loss function.
    val convolution_kernel = (repeated_times - index_times)
      .abs
      .pow(p)
      .divide(p.multiply(Tensor(-1.0).castTo[P].toOutput))
      .divide(time_scale.square)
      .exp

    //Convolve the kernel with the loss tensor, yielding the weighted loss tensor
    val weighted_loss_tensor =
      (repeated_preds - targets.castTo[P])
        .square
        .multiply(convolution_kernel)
        .sum(axes = 1)
        .divide(convolution_kernel.sum(axes = 1))
        .mean()

    /*
     * Compute the prior term, which is a softplus
     * function applied on an affine transformation
     * of the empirical Spearman Correlation between
     * the predicted targets and unscaled time lags.
     * */
    val prior =
      norm_ranked_targets.multiply(norm_ranked_unsc_lags)
        .mean()
        .subtract(Tensor(corr_cutoff).castTo[P].toOutput)
        .multiply(Tensor(prior_scaling).castTo[P].toOutput)
        .softplus
        .multiply(Tensor(prior_weight).castTo[P].toOutput)

    weighted_loss_tensor.add(prior).castTo[L]
  }
}
