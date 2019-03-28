package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * <h3>Weighted L2 Loss Function</h3>
  * */
case class WeightedL2FluxLoss[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric: IsNotQuantized,
L: TF : IsFloatOrDouble](
  override val name: String)
  extends Loss[(Output[P], Output[T]), L](name) {

  override val layerType: String = "WeightedL2FluxLoss"

  override def forwardWithoutContext(input: (Output[P], Output[T]))(implicit mode: Mode): Output[L] =
    ops.NN.l2Loss((input._1 - input._2.castTo[P])*input._2.castTo[P].sigmoid.sqrt, name).castTo[L]
}

