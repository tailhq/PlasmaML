package io.github.mandar2812.PlasmaML.helios.core

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.kernels.StationaryKernel
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>Kernel Weighted Solar Wind Loss (KSW Loss)</h3>
  *
  * A weighted loss function which enables fuzzy learning of
  * the solar wind propagation from heliospheric
  * images to ACE.
  *
  * @author mandar2812
  * */
class KernelWeightedSWLoss(
  override val name: String,
  val kernel: StationaryKernel[Double, Double, DenseMatrix[Double]])
  extends Loss[(Output, Output)](name) {

  override val layerType: String = "KernelWeightedSWLoss"

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {
    ops.NN.l2Loss(input._1 - input._2, name = name)
  }
}
