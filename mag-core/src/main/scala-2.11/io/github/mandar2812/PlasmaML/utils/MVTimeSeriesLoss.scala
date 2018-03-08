package io.github.mandar2812.PlasmaML.utils

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Loss
import org.platanios.tensorflow.api.ops.Output

//TODO: Transfer file to DynaML
case class MVTimeSeriesLoss(override val name: String)
  extends Loss[(Output, Output)](name) {
  override val layerType: String = "L2Loss"

  override protected def _forward(input: (Output, Output), mode: Mode): Output = {
    input._1.subtract(input._2).square.mean(axes = 0).sum()
  }
}