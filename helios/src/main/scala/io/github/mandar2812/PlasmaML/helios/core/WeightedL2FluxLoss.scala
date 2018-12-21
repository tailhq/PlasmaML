package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>Weighted L2 Loss Function</h3>
  * */
case class WeightedL2FluxLoss(
  override val name: String)
  extends Loss[(Output, Output)](name) {

  override val layerType: String = "WeightedL2FluxLoss"

  override protected def _forward(input: (Output, Output))(implicit mode: Mode): Output =
    if (input._2.dataType == FLOAT32)
      ops.NN.l2Loss((input._1 - input._2)*input._2.sigmoid.sqrt, name = name)
    else ops.NN.l2Loss((input._1 - input._2)*input._2.cast(FLOAT32).sigmoid.sqrt, name = name)
}

