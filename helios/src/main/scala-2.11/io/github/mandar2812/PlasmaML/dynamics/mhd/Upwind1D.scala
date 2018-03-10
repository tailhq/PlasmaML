package io.github.mandar2812.PlasmaML.dynamics.mhd

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.models.neuralnets.{Activation, LazyNeuralStack, NeuralLayer, VectorLinear}
import io.github.mandar2812.dynaml.pipes._
import org.apache.log4j.Logger
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tensors.Tensor

/**
  * Implementation of the Upwind 1-dimensional
  * solar wind propagation model.
  *
  * This is a discretized version of the 1-d simplified
  * MHD equations for the solar wind. These equations reduce
  * to an inviscid Burgers system.
  * */
class Upwind1D(
  rDomain: (Double, Double),
  nR: Int, nTheta: Int,
  omega_rot: Double) {

  /**
    * The Carrington longitude lies
    * between 0 and 2*Pi radians.
    * */
  val thetaDomain: (Double, Double) = (0d, 2*math.Pi)

  val (deltaR, deltaTheta) = (
    (rDomain._2 - rDomain._1)/nR,
    (thetaDomain._2 - thetaDomain._1)/nTheta
  )

  val stencil: (Seq[Double], Seq[Double]) = Upwind1D.buildStencil(rDomain, nR, thetaDomain, nTheta)

  protected val upwindForwardLayer: NeuralLayer[(Double, Double, Double), Tensor, Tensor] =
    NeuralLayer[(Double, Double, Double), Tensor, Tensor](
      Upwind1D.forwardProp(nTheta),
      Activation(
        (x: Tensor) => x,
        (x: Tensor) => dtf.fill(FLOAT32, x.shape.entriesIterator.map(_.asInstanceOf[Int]).toSeq:_*)(1d)
      )
    )(
      (omega_rot, deltaR, deltaTheta)
    )

  val computationalStack = LazyNeuralStack(_ => upwindForwardLayer, nR)

  def solve(v0: Tensor): Stream[Tensor] = computationalStack forwardPropagate v0

}

object Upwind1D {

  private val logger = Logger.getLogger(this.getClass)

  def forwardProp(nTheta: Int) = MetaPipe((params: (Double, Double, Double)) => (v: Tensor) => {
    val (omega, dR, dT) = params
    val invV = v.pow(-1d)

    val forwardDiffMat = dtf.tensor_f64(nTheta, nTheta)(
      DenseMatrix.tabulate(nTheta, nTheta)((i, j) => if(i == j) -1d else if(j == (i+1)%nTheta) 1d else 0d).t.toArray:_*
    )

    v.add(forwardDiffMat.matmul(v).multiply(invV).multiply((dR/dT)*omega))
    //v + (dR/dT)*omega*invV*:*(forwardDiffMat*v)
  })

  def windWithLag(
    nR: Int, nTheta: Int,
    deltaR: Double, deltaTheta: Double): DataPipe[Stream[Tensor], (Double, Float)] =
    DataPipe((result: Stream[Tensor]) => {
      val sliding_avg =
        DenseMatrix.tabulate(nR, nR + 1)(
          (i, j) => if(i == j) 0.5d else if(j == (i+1)) 0.5d else 0.0d)

      val velocities = DenseVector(result.map(_(0).scalar.asInstanceOf[Double]):_*)

      val deltat = sum((sliding_avg*velocities).map(deltaR/_))

      val v = velocities(-1)
      (deltat, v.toFloat)
    })

  def buildStencil(
    rDomain: (Double, Double), nR: Int,
    thetaDomain: (Double, Double), nTheta: Int,
    logScaleFlags: (Boolean, Boolean) = (false, false)): (Seq[Double], Seq[Double]) = {

    logger.info("----------------------------------")
    logger.info("Domain stencil: \n")

    val deltaR =
      if(logScaleFlags._1) math.log(rDomain._2 - rDomain._1)/nR
      else (rDomain._2 - rDomain._1)/nR

    val deltaTheta =
      if(logScaleFlags._2) math.log(thetaDomain._2 - thetaDomain._1)/nTheta
      else (thetaDomain._2 - thetaDomain._1)/nTheta

    logger.info("Radial Distance")
    if(logScaleFlags._1) logger.info("Logarithmic Scale")
    logger.info(rDomain._1+" =< L =< "+rDomain._2)
    logger.info("Δr = "+deltaR+"\n")

    logger.info("Carrington Longitude")
    if(logScaleFlags._2) logger.info("Logarithmic Scale")
    logger.info(thetaDomain._1+" =< t =< "+thetaDomain._2)
    logger.info("Δ"+0x03B8.toChar+" = "+deltaTheta)
    logger.info("----------------------------------")


    val rVec = if(logScaleFlags._1) {
      Seq.tabulate[Double](nR+1)(i =>
        if(i == 0) rDomain._1
        else if(i < nR) rDomain._1+math.exp(deltaR*i)
        else rDomain._2)
    } else {
      Seq.tabulate[Double](nR+1)(i =>
        if(i < nR) rDomain._1+(deltaR*i)
        else rDomain._2)
    }

    val thetaVec = if(logScaleFlags._2) {
      Seq.tabulate[Double](nTheta+1)(i =>
        if(i ==0) thetaDomain._1
        else if(i < nTheta) thetaDomain._1+math.exp(deltaTheta*i)
        else thetaDomain._2)
    } else {
      Seq.tabulate[Double](nTheta+1)(i =>
        if(i < nTheta) thetaDomain._1+(deltaTheta*i)
        else thetaDomain._2)
    }

    (rVec, thetaVec)
  }

}