package io.github.mandar2812.PlasmaML.dynamics.mhd

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.tensorflow.dtf
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, Initializer, RandomUniformInitializer}

/**
  * <h3>Upwind Solar Wind Model</h3>
  *
  * 1-Dimensional <i>up-wind</i> solar wind
  * propagation model, implemented as a tensorflow
  * computational layer.
  *
  * @param rDomain The lower and upper limits of the radial domain.
  * @param nR Number of divisions of the radial domain.
  * @param omegaInit Initialiser for the solar rotation speed.
  * */
case class UpwindTF[P: TF: IsFloatOrDouble](
  override val name: String,
  rDomain: (Double, Double), nR: Int,
  vmin: Double = 200, vmax: Double = 1200,
  omegaInit: Initializer = RandomUniformInitializer(0.01f, 1.0f))
  extends Layer[Output[P], Output[P]](name) {

  override val layerType: String = s"Upwind1D[r:$rDomain, nR:$nR]"

  protected val (beta, gamma) = (
    Tensor(vmax.toFloat).subtract(vmin.toFloat).castTo[P],
    Tensor(vmin.toFloat).castTo[P])

  /**
    * The Carrington longitude lies
    * between 0 and 2*Pi radians.
    * */
  val thetaDomain: (Double, Double) = (0d, 2*math.Pi)

  protected val deltaR: Float = (rDomain._2.toFloat - rDomain._1.toFloat)/nR

  override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): Output[P] = {

    val nTheta = input.shape(-1)

    val deltaTheta: Float = (thetaDomain._2.toFloat - thetaDomain._1.toFloat)/nTheta

    val dv_dtheta = tf.constant[P](
      dtf.tensor_f64(nTheta, nTheta)(DenseMatrix.tabulate(nTheta, nTheta)(
        (i, j) => if(i == j) -1.0 else if(j == (i+1)%nTheta) 1.0 else 0.0).t.toArray:_*).castTo[P],
      Shape(nTheta, nTheta),
      "DeltaV")

    //Rotational speed of the sun, as a trainable parameter
    val omega_rot = tf.variable[P]("OmegaRot", Shape(), omegaInit)
    val alpha     = tf.variable[P]("alpha", Shape(), ConstantInitializer(0.15))

    //Propagate solar wind, from r = rmin to r = rmax
    val velocity_profile = tf.stack(
      (1 to nR).scanLeft(input)((x, _) => {

        val invV = x.multiply(beta).add(gamma).pow(Tensor(-1f).castTo[P])

        x.add(x.tensorDot(dv_dtheta, Seq(1), Seq(0))
          .multiply(invV)
          .multiply(omega_rot)
          .multiply(Tensor(deltaR/deltaTheta).castTo[P]))
      }),
      axis = -1)

    //Compute residual acceleration
    val rH = 50.0

    val r = tf.constant[P](
      dtf.tensor_f64(nR + 1)(
        utils.range(rDomain._1, rDomain._2, nR) :+ rDomain._2 :_*
      ).divide(rH).castTo[P],
      Shape(nR + 1))

    val v_acc = tf
      .stack(Seq.fill(nR + 1)(input.multiply(alpha)), axis = -1)
      .multiply(
        r.multiply(Tensor(-1.0).castTo[P])
          .exp.multiply(Tensor(-1.0).castTo[P])
          .add(Tensor(1.0).castTo[P])
      )

    velocity_profile.add(v_acc)
  }
}

case class UpwindPropogate[P: TF: IsFloatOrDouble](
  override val name: String,
  rDomain: (Double, Double),
  nR: Int, nTheta: Int,
  omegaInit: Initializer = RandomUniformInitializer())
  extends Layer[Output[P], Output[P]](name) {

  override val layerType: String = s"UpwindPropogate[r:$rDomain, nR:$nR, nTheta:$nTheta]"

  /**
    * The Carrington longitude lies
    * between 0 and 2*Pi radians.
    * */
  val thetaDomain: (Double, Double) = (0d, 2*math.Pi)

  protected val (deltaR, deltaTheta) = (
    (rDomain._2 - rDomain._1)/nR,
    (thetaDomain._2 - thetaDomain._1)/nTheta
  )

  override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): Output[P] = {

    val velocities = input(::, 0, ::)

    val sliding_avg_tensor = dtf.tensor_f64(
      nR, nR + 1)(
      DenseMatrix.tabulate(nR, nR + 1)(
        (i, j) => if(i == j) 0.5 else if(j == (i+1)) 0.5 else 0.0
      ).t.toArray:_*
    ).castTo[P]

    val sliding_avg = tf.constant[P](
      sliding_avg_tensor,
      Shape(nR, nR+1), "VAvgOp")

    val deltat = velocities.tensorDot(sliding_avg, Seq(1), Seq(0))
      .pow(Tensor(-1d).castTo[P])
      .multiply(Tensor(deltaR).castTo[P])
      .sum[Int]()

    val v = input(::, 0, -1).reshape(Shape())

    tf.stack(Seq(v, deltat), axis = -1).reshape(Shape(2))
  }
}
