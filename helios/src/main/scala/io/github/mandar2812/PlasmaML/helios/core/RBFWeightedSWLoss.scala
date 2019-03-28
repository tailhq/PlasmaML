package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, Loss}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>RBF-Kernel Weighted Solar Wind Loss (KSW Loss)</h3>
  *
  * A weighted loss function which enables fuzzy learning of
  * the solar wind propagation from heliospheric
  * images to ACE.
  *
  * @param name Unique string identifier of the loss/layer
  * @param size_causal_window The size of the finite time horizon, over
  *                           which time-lag and signal predictions are made.
  * @param kernel_time_scale  The time scale of the Radial Basis Function (RBF) convolution
  *                           kernel, defaults to 1.
  * @param kernel_norm_exponent The Lp norm used in the kernel, defaults to 2
  * @param corr_cutoff The cut-off applied to the Spearman correlation between
  *                    the predicted targets and time lags, defaults to -0.75.
  * @param prior_scaling A scaling parameter multiplier applied in the prior term, defaults to 1.
  * @param batch The batch size for each training epoch.
  * @author mandar2812
  * */
case class RBFWeightedSWLoss[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric: IsNotQuantized,
L: TF : IsFloatOrDouble](
  override val name:    String,
  size_causal_window:   Int,
  kernel_time_scale:    Double = 3d,
  kernel_norm_exponent: Double = 2,
  corr_cutoff:          Double = -0.75,
  prior_scaling:        Double = 1.0,
  batch:                Int    = -1,
  scale_lags: Boolean = true)
  extends Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType: String = s"RBFSW[horizon:$size_causal_window, timescale:$kernel_time_scale]"

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    val predictions   = input._1._1
    val timelags      = input._1._2
    val targets       = input._2


    //Determine the batch size, if not provided @TODO: Iron out bugs in this segment
    val batchSize =
      if(batch == -1) predictions.size.toInt/predictions.shape.toTensor.prod[Int]().scalar.asInstanceOf[Int]
      else batch

    //Sort the predicted targets and time lags, obtaining
    //the ranks of each element.
    val rank_preds = predictions.topK(batchSize)._2.castTo[P]
    val rank_unsc_lags = timelags.topK(batchSize)._2.castTo[P]

    //Standardize (mean center) the ranked tensors
    val (ranked_preds_mean, ranked_unscaled_lags_mean) = (rank_preds.mean(), rank_unsc_lags.mean())

    val (ranked_preds_std, ranked_unscaled_lags_std)   = (
      rank_preds.subtract(ranked_preds_mean).square.mean().sqrt,
      rank_unsc_lags.subtract(ranked_unscaled_lags_mean).square.mean().sqrt)

    val (norm_ranked_targets, norm_ranked_unsc_lags)   = (
      rank_preds.subtract(ranked_preds_mean).divide(ranked_preds_std),
      rank_unsc_lags.subtract(ranked_unscaled_lags_mean).divide(ranked_unscaled_lags_std))


    val repeated_times      = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val repeated_preds      = tf.stack(Seq.fill(size_causal_window)(predictions), axis = -1)

    val index_times: Output[P] = Tensor(
      (0 until size_causal_window).map(_.toDouble)
    ).reshape(
      Shape(size_causal_window)
    ).castTo[P].toOutput

    //Calculate the convolution kernel of the loss function.
    val convolution_kernel = repeated_times.subtract(index_times)
      .abs
      .pow(Tensor(kernel_norm_exponent).castTo[P].toOutput)
      .multiply(Tensor(-1.0).castTo[P]/Tensor(kernel_norm_exponent).castTo[P])
      .divide(Tensor(kernel_time_scale).castTo[P])
      .exp

    val target_err = repeated_preds.subtract(targets.castTo[P])

    //Convolve the kernel with the loss tensor, yielding the weighted loss tensor
    val weighted_loss_tensor = target_err
      .square
      .multiply(convolution_kernel)
      .sum(axes = 1)
      .divide(convolution_kernel.sum(axes = 1))
      .mean()

    /*
    * Compute the prior term, which is an affine transformation
    * of the empirical Spearman Correlation between
    * the predicted targets and unscaled time lags.
    * */
    val prior =
      norm_ranked_targets.multiply(norm_ranked_unsc_lags)
        .mean()
        .subtract(Tensor(corr_cutoff).castTo[P].toOutput)
        .multiply(Tensor(prior_scaling).castTo[P].toOutput)

    val offset: Double = (1.0 - math.abs(corr_cutoff))*prior_scaling

    weighted_loss_tensor.add(prior).add(Tensor(offset).castTo[P].toOutput).castTo[L]
  }
}


object RBFWeightedSWLoss {

  def output_mapping[P: TF: IsFloatOrDouble](
    name: String,
    size_causal_window: Int,
    kernel_time_scale: Double = 3d,
    scale_lags: Boolean = true): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {

      override val layerType: String = s"OutputRBFSW[horizon:$size_causal_window, timescale:$kernel_time_scale]"

      private[this] val scaling = Tensor(size_causal_window.toDouble-1d).castTo[P]

      val alpha = Tensor(1.0).castTo[P].toOutput
      val nu    = Tensor(1.0).castTo[P].toOutput
      val q     = Tensor(1.0).castTo[P].toOutput

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {

        val lags = if (scale_lags) {
          input(::, -1)
            .multiply(alpha.add(Tensor(1E-6).castTo[P].toOutput).square.multiply(Tensor(-1.0).castTo[P].toOutput))
            .exp
            .multiply(q.square)
            .add(Tensor(1.0).castTo[P].toOutput)
            .pow(nu.square.pow(Tensor(-1.0).castTo[P].toOutput).multiply(Tensor(-1.0).castTo[P].toOutput))
            .multiply(scaling.toOutput)
        } else {
          input(::, -1)
        }

        (input(::, 0), lags)
      }
    }

}