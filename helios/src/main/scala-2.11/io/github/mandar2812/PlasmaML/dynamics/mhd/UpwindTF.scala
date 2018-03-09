package io.github.mandar2812.PlasmaML.dynamics.mhd

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.tensorflow.dtf
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomUniformInitializer}

case class UpwindTF(
  override val name: String, rDomain: (Double, Double),
  nR: Int, nTheta: Int, omegaInit: Initializer = RandomUniformInitializer()) extends Layer[Output, Output](name) {

  override val layerType: String = "Upwind1D"

  /**
    * The Carrington longitude lies
    * between 0 and 2*Pi radians.
    * */
  val thetaDomain: (Double, Double) = (0d, 2*math.Pi)

  protected val (deltaR, deltaTheta) = (
    (rDomain._2 - rDomain._1)/nR,
    (thetaDomain._2 - thetaDomain._1)/nTheta
  )

  val dv_dtheta = dtf.tensor_f32(nTheta, nTheta)(
    DenseMatrix.tabulate(nTheta, nTheta)(
      (i, j) => if(i == j) -1.0 else if(j == (i+1)%nTheta) 1.0 else 0.0).t.toArray:_*
  ).toOutput

  override protected def _forward(input: Output, mode: Mode): Output = {

    val omega_rot      = tf.variable("OmegaRot", input.dataType, Shape(), omegaInit)

    tf.stack(
      (1 to nR).scanLeft(input)((x, _) => {
        val invV = x.pow(-1d)
          x.add(dv_dtheta.matmul(x).multiply(invV).multiply(omega_rot).multiply(deltaR/deltaTheta))
      }),
      axis = -1)

  }
}

case class UpwindPropogate(
  override val name: String, rDomain: (Double, Double),
  nR: Int, nTheta: Int, omegaInit: Initializer = RandomUniformInitializer()) extends Layer[Output, Output](name) {
  override val layerType: String = "UpwindPropogate"

  /**
    * The Carrington longitude lies
    * between 0 and 2*Pi radians.
    * */
  val thetaDomain: (Double, Double) = (0d, 2*math.Pi)

  protected val (deltaR, deltaTheta) = (
    (rDomain._2 - rDomain._1)/nR,
    (thetaDomain._2 - thetaDomain._1)/nTheta
  )

  val sliding_avg = dtf.tensor_f32(nR, nR + 1)(
    DenseMatrix.tabulate(nR, nR + 1)(
      (i, j) => if(i == j) 0.5 else if(j == (i+1)) 0.5 else 0.0).t.toArray:_*
  ).toOutput

  override protected def _forward(input: Output, mode: Mode): Output = {

    val velocities = input(::, 0, ::).reshape(Shape(nR, 1))

    val deltat = sliding_avg.matmul(velocities).pow(-1d).multiply(deltaR).sum()

    val v = input(::, 0, -1).reshape(Shape())

    tf.stack(Seq(v, deltat), axis = -1)
  }
}
