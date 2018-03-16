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
case class RBFWeightedSWLoss(
  override val name: String,
  size_causal_window: Int,
  time_scale: Double = 3d)
  extends Loss[(Output, Output)](name) {

  override val layerType: String = s"RBFSW[horizon:$size_causal_window, timescale:$time_scale]"

  private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {

    //Obtain section corresponding to velocity predictions
    val predictions = input._1(::, 0)

    val targets = input._2

    val timelags = input._1(::, 1).sigmoid.multiply(scaling).floor

    val repeated_times = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)
    
    val repeated_preds = tf.stack(Seq.fill(size_causal_window)(predictions), axis = -1)
    
    val index_times: Output = Tensor((0 until size_causal_window).map(_.toDouble)).reshape(Shape(size_causal_window))


    val convolution_kernel = repeated_times.subtract(index_times)
      .square
      .multiply(-0.5)
      .divide(time_scale)
      .exp

    val weighted_loss_tensor = repeated_preds.subtract(targets)
        .square
        .multiply(convolution_kernel)
        .sum(axes = 1)
        .divide(convolution_kernel.sum(axes = 1))
        .mean()

    weighted_loss_tensor
  }
}
