package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, Loss}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>RBF-Kernel Weighted Solar Wind Loss (KSW Loss)</h3>
  * <h4>avec dynamic time scale prediction</h4>
  *
  * A weighted loss function which enables fuzzy learning of
  * the solar wind propagation from heliospheric
  * images to ACE.
  *
  * @author mandar2812
  * */
case class DynamicRBFSWLoss[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric: IsNotQuantized,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int) extends
  Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType: String = s"DynamicRBFSW[$size_causal_window]"

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    //Obtain section corresponding to velocity predictions
    val preds = input._1._1

    val times_and_scales = input._1._2

    val times = times_and_scales(::, 0)

    val timescales = times_and_scales(::, 1)

    val repeated_timescales = tf.stack(Seq.fill(size_causal_window)(timescales), axis = -1)

    val repeated_times = tf.stack(Seq.fill(size_causal_window)(times), axis = -1)

    val index_times = Tensor((0 until size_causal_window).map(_.toDouble))
      .reshape(Shape(1, size_causal_window))
      .castTo[P]

    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(preds), axis = -1)

    val convolution_kernel =
      (repeated_times - index_times)
        .abs
        .multiply(Tensor(-1d).castTo[P].toOutput)
        .divide(repeated_timescales)
        .exp

    val weighted_loss_tensor =
      (repeated_preds - input._2.castTo[P])
        .square
        .multiply(convolution_kernel)
        .sum(axes = 1)
        .divide(convolution_kernel.sum(axes = 1))
        .mean()

    weighted_loss_tensor.castTo[L]
  }
}

object DynamicRBFSWLoss {

  def output_mapping[P: TF: IsFloatOrDouble](
    name: String, size_causal_window: Int): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {

      private[this] val scaling = Tensor(size_causal_window.toDouble-1d).castTo[P].toOutput

      override val layerType: String = s"OutputDynamicRBFSW[$size_causal_window]"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {
        (input(::, 0), tf.concatenate(Seq(input(::, 1).sigmoid.multiply(scaling), input(::, 2).exp), axis = 1))
      }
    }

}
