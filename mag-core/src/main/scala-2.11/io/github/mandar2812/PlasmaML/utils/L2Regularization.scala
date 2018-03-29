package io.github.mandar2812.PlasmaML.utils

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{ReuseOrCreateNew, ReuseExistingOnly}
import org.platanios.tensorflow.api.types.DataType

case class L2Regularization(
  names: Seq[String],
  dataTypes: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output, Output]("") {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  override protected def _forward(input: Output, mode: Mode): Output = {

    val weights = names.zip(dataTypes.zip(shapes)).map(n =>
      tf.variable(n._1, dataType = DataType.fromName(n._2._1), shape = n._2._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.square.sum()).reduce(_.add(_)).multiply(0.5*reg)

    input.add(reg_term)
  }
}

case class L1Regularization(names: Seq[String], dataTypes: Seq[String], shapes: Seq[Shape], reg: Double = 0.01) extends
  Layer[Output, Output]("") {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  override protected def _forward(input: Output, mode: Mode): Output = {

    val weights = names.zip(dataTypes.zip(shapes)).map(n =>
      tf.variable(n._1, dataType = DataType.fromName(n._2._1), shape = n._2._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.abs.sum()).reduce(_.add(_)).multiply(reg)

    input.add(reg_term)
  }
}
