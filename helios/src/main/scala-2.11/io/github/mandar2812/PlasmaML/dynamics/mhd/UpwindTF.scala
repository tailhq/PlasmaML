package io.github.mandar2812.PlasmaML.dynamics.mhd

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.tensorflow.dtf
import org.platanios.tensorflow.api.{Output, Shape, tf}
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

  override protected def _forward(input: Output, mode: Mode): Output = {

    val omega_rot      = tf.variable("OmegaRot", input.dataType, Shape(), omegaInit)

    val invV = input.pow(-1d)

    val forwardDiffMat = dtf.tensor_f32(nTheta, nTheta)(
      DenseMatrix.tabulate(nTheta, nTheta)(
        (i, j) => if(i == j) -1.0 else if(j == (i+1)%nTheta) 1.0 else 0.0).t.toArray:_*
    ).toOutput

    input.add(forwardDiffMat.matmul(input).multiply(invV).multiply(omega_rot).multiply(deltaR/deltaTheta))
  }
}
