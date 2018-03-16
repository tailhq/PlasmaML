package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * Stacks output produced by a tensorflow concatenation layer
  *
  * @param axis The axis over which the tensors should be concatenated,
  *             defaults to -1
  *
  * @author mandar2812 date 2018/03/16
  * */
//TODO: Transfer to DynaML when finished.
case class Stack(override val name: String, axis: Int = -1) extends Layer[Seq[Output], Output](name) {

  override val layerType: String = s"Stack[axis:$axis]"

  override protected def _forward(input: Seq[Output], mode: Mode): Output = tf.stack(input, axis)

  override def forward(input: Seq[Output], mode: Mode): Output = tf.stack(input, axis)
}


case class IdentityLayer[I](override val name: String) extends Layer[I, I](name) {

  override val layerType: String = s"Identity"

  override protected def _forward(input: I, mode: Mode): I = input


}