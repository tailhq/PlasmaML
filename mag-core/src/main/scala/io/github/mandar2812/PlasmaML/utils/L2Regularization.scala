package io.github.mandar2812.PlasmaML.utils

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, IsReal, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.ReuseExistingOnly



case class L2Regularization[D: TF: IsNotQuantized](
  override val name: String,
  names: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output[D], Output[D]](name) {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  @throws[IllegalArgumentException]
  @throws[ShapeMismatchException]
  @throws[InvalidDataTypeException]
  override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Output[D] = {

    val weights = names.zip(shapes).map(n =>
      tf.variable[D](n._1, shape = n._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.square.sum[Int]()).reduce(_.add(_)).multiply(Tensor(0.5*reg).toOutput.castTo[D])

    input.add(reg_term)
  }
}


case class L1Regularization[D: TF: IsNotQuantized: IsReal](
  override val name: String,
  names: Seq[String],
  shapes: Seq[Shape],
  reg: Double = 0.01) extends
  Layer[Output[D], Output[D]](name) {

  override val layerType: String = s"L2Reg[gamma:$reg]"

  @throws[IllegalArgumentException]
  @throws[ShapeMismatchException]
  @throws[InvalidDataTypeException]
  override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Output[D] = {

    val weights = names.zip(shapes).map(n =>
      tf.variable[D](n._1, shape = n._2, reuse = ReuseExistingOnly)
    )

    val reg_term = weights.map(_.abs.sum[Int]()).reduce(_.add(_)).multiply(Tensor(reg).toOutput.castTo[D])

    input.add(reg_term)
  }
}