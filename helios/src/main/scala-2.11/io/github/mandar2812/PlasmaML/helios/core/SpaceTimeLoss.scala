package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, Loss}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.{::, FLOAT32, Shape, Tensor, tf}

case class SpaceTimeLoss(
  override val name: String,
  size_causal_window:   Int,
  corr_cutoff:          Double  = -0.75,
  prior_weight:         Double  = 1.0,
  prior_scaling:        Double  = 1.0,
  batch:                Int     = -1,
  scale_lags:           Boolean = true) extends
  Loss[((Output, Output), Output)](name) {

  override val layerType: String = s"SpaceTimeLoss[horizon:$size_causal_window]"

  private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

  override protected def _forward(input: ((Output, Output), Output))(implicit mode: Mode): Output = {

    val predictions   = input._1._1
    val unscaled_lags = input._1._2
    val targets       = input._2

    //Determine the batch size, if not provided @TODO: Iron out bugs in this segment
    val batchSize =
      if(batch == -1) predictions.size.toInt/predictions.shape.toTensor().prod().scalar.asInstanceOf[Int]
      else batch


    //Perform scaling of time lags using the generalized logistic curve.
    //First define some parameters.

    //val alpha: tf.Variable = tf.variable("alpha", FLOAT32, Shape(), tf.RandomUniformInitializer())
    //val nu:    tf.Variable = tf.variable("nu",    FLOAT32, Shape(), tf.OnesInitializer)
    //val q:     tf.Variable = tf.variable("Q",     FLOAT32, Shape(), tf.OnesInitializer)

    val alpha = Tensor(0.5)
    val nu = Tensor(1.0)
    val q = Tensor(1.0)

    val timelags           = unscaled_lags/*if (scale_lags) {
      unscaled_lags
        .multiply(alpha.add(1E-6).square.multiply(-1.0))
        .exp
        .multiply(q.square)
        .add(1.0)
        .pow(nu.square.pow(-1.0).multiply(-1.0))
        .multiply(scaling)
        .floor
    } else {
      unscaled_lags.floor
    }
*/
    val repeated_times = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(predictions), axis = -1)

    val index_times: Output = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(size_causal_window))


    val time_loss = repeated_times.subtract(index_times)
      .square.multiply(0.5)

    val space_loss = repeated_preds.subtract(targets).square.multiply(0.5)

    //Sort the predicted targets and time lags, obtaining
    //the ranks of each element.
    val rank_preds = predictions.topK(batchSize)._2.cast(FLOAT32)
    val rank_unsc_lags = timelags.topK(batchSize)._2.cast(FLOAT32)

    //Standardize (mean center) the ranked tensors
    val (ranked_preds_mean, ranked_unscaled_lags_mean) = (rank_preds.mean(), rank_unsc_lags.mean())

    val (ranked_preds_std, ranked_unscaled_lags_std)   = (
      rank_preds.subtract(ranked_preds_mean).square.mean().sqrt,
      rank_unsc_lags.subtract(ranked_unscaled_lags_mean).square.mean().sqrt)

    val (norm_ranked_targets, norm_ranked_unsc_lags)   = (
      rank_preds.subtract(ranked_preds_mean).divide(ranked_preds_std),
      rank_unsc_lags.subtract(ranked_unscaled_lags_mean).divide(ranked_unscaled_lags_std))

    /*
    * Compute the prior term, which is an affine transformation
    * of the empirical Spearman Correlation between
    * the predicted targets and unscaled time lags.
    * */
    val prior =
      norm_ranked_targets.multiply(norm_ranked_unsc_lags)
        .mean()
        .subtract(corr_cutoff)
        .multiply(prior_scaling)

    val offset: Double = (1.0 - math.abs(corr_cutoff))*prior_scaling

    space_loss.add(time_loss).sum(axes = 1).mean().add(prior).add(offset)
  }
}

object SpaceTimeLoss {

  def output_mapping(
    name: String,
    size_causal_window: Int,
    scale_lags: Boolean = true): Layer[Output, (Output, Output)] =
    new Layer[Output, (Output, Output)](name) {

      override val layerType: String = s"OutputSpaceTimeLoss[horizon:$size_causal_window]"

      private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

      val alpha = Tensor(1.0)
      val nu    = Tensor(1.0)
      val q     = Tensor(1.0)

      override protected def _forward(input: Output)(implicit mode: Mode): (Output, Output) = {

        val lags = if (scale_lags) {
          input(::, -1)
            .multiply(alpha.add(1E-6).square.multiply(-1.0))
            .exp
            .multiply(q.square)
            .add(1.0)
            .pow(nu.square.pow(-1.0).multiply(-1.0))
            .multiply(scaling)
        } else {
          input(::, -1)
        }

        (input(::, 0), lags)
      }
    }

}