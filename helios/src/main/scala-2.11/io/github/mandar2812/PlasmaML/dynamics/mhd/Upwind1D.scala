package io.github.mandar2812.PlasmaML.dynamics.mhd

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.neuralnets.{LazyNeuralStack, NeuralLayer, VectorLinear}
import io.github.mandar2812.dynaml.pipes.MetaPipe
import org.apache.log4j.Logger

/**
  * Implementation of the Upwind 1-dimensional
  * solar wind propagation model.
  *
  * This is a discretized version of the 1-d simplified
  * MHD equations for the solar wind. These equations reduce
  * to an inviscid Burgers system.
  * */
class Upwind1D(
  rDomain: (Double, Double), nR: Int,
  thetaDomain: (Double, Double), nTheta: Int,
  omega_rot: Double) {

  protected val (deltaR, deltaTheta) = (
    (rDomain._2 - rDomain._1)/nR,
    (thetaDomain._2 - thetaDomain._1)/nTheta
  )

  val stencil: (Seq[Double], Seq[Double]) = Upwind1D.buildStencil(rDomain, nR, thetaDomain, nTheta)

  protected val upwindForwardLayer: NeuralLayer[(Double, Double, Double), DenseVector[Double], DenseVector[Double]] =
    NeuralLayer(
      Upwind1D.forwardProp(nTheta), VectorLinear)(
      (omega_rot, deltaR, deltaTheta))

  val computationalStack = LazyNeuralStack(_ => upwindForwardLayer, nR)

  def solve(v0: DenseVector[Double]): Stream[DenseVector[Double]] = computationalStack forwardPropagate v0

}

object Upwind1D {

  private val logger = Logger.getLogger(this.getClass)

  def forwardProp(nTheta: Int) = MetaPipe((params: (Double, Double, Double)) => (v: DenseVector[Double]) => {
    val (omega, dR, dT) = params
    val invV = v.map(1/_)

    val forwardDiffMat = DenseMatrix.tabulate(nTheta, nTheta)((i, j) =>
      if(i == j) -1d else if(j == (i+1)%nTheta) 1d else 0d)

    v + (dR/dT)*omega*invV*:*(forwardDiffMat*v)
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