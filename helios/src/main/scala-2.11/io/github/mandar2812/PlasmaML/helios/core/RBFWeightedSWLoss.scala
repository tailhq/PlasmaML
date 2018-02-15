package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>RBF-Kernel Weighted Solar Wind Loss (KSW Loss)</h3>
  *
  * A weighted loss function which enables fuzzy learning of
  * the solar wind propagation from heliospheric
  * images to ACE.
  *
  * @author mandar2812
  * */
class RBFWeightedSWLoss(
  override val name: String,
  val size_causal_window: Int,
  val time_scale: tf.Variable = tf.variable("time_scale", FLOAT32, Shape(), tf.OnesInitializer))
  extends Loss[(Output, Output)](name) {

  override val layerType: String = "KernelWeightedSWLoss"

  private[this] val scaling = Tensor(size_causal_window.toDouble)

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    //Obtain section corresponding to velocity predictions
    val preds = input._1(::, 0)

    val times = input._1(::, 1).sigmoid.multiply(scaling)

    val size_batch = input._1.shape(0)

    val repeated_times = tf.stack(Seq.fill(size_causal_window)(times), axis = -1)

    val index_times = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(1, size_causal_window))

    val repeated_index_times = tf.stack(Seq.fill(size_batch)(index_times), axis = 0)

    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(preds), axis = -1)

    val convolution_kernel = (repeated_index_times - repeated_times).square.multiply(-0.5).divide(time_scale.value).exp

    val weighted_loss_tensor =
      (repeated_preds - input._2)
        .multiply(convolution_kernel)
        .sum(axes = 1)
        .divide(convolution_kernel.sum(axes = 1, keepDims = true))

    ops.NN.l2Loss(weighted_loss_tensor, name = name)
  }
}
