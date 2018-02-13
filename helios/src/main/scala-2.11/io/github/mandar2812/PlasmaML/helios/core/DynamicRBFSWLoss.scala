package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
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
class DynamicRBFSWLoss(
  override val name: String,
  val size_causal_window: Int) extends Loss[(Output, Output)](name) {

  override val layerType: String = "KernelWeightedSWLoss"

  private[this] val scaling = Tensor(size_causal_window.toDouble)

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    //Obtain section corresponding to velocity predictions
    val preds = input._1(::, 0)

    val times = input._1(::, 1).sigmoid.multiply(scaling)

    val timescales = input._1(::, 2).square

    val size_batch = input._1.shape(0)

    val repeated_timescales = tf.stack(Seq.fill(size_causal_window)(timescales), axis = -1)

    val repeated_times = tf.stack(Seq.fill(size_causal_window)(times), axis = -1)

    val index_times = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(1, size_causal_window))

    val repeated_index_times = tf.stack(Seq.fill(size_batch)(index_times), axis = 0)

    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(preds), axis = -1)

    val convolution_kernel =
      (repeated_index_times - repeated_times)
        .square.multiply(-0.5)
        .divide(repeated_timescales)
        .exp

    val weighted_loss_tensor = (repeated_preds - input._2).multiply(convolution_kernel).sum(axes = 1)

    ops.NN.l2Loss(weighted_loss_tensor, name = name)
  }
}
