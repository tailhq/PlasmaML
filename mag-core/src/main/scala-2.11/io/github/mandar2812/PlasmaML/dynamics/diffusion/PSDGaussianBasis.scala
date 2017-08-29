package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.analysis.{Basis, RadialBasis}
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.pipes.{Encoder, TupleIntegerEncoder}

/**
  * <h3>Phase Space Density: Mesh-Radial Basis<h3>
  *
  * Implements a radial basis expansion for the phase space density
  * with nodes placed on a regular space time mesh.
  *
  * */
class PSDGaussianBasis(
  val lShellLimits: (Double, Double), val nL: Int,
  val timeLimits: (Double, Double), val nT: Int) extends
  Basis[(Double, Double)] {

  var mult = 4d

  val (lSeq, tSeq) = RadialDiffusion.buildStencil(
    lShellLimits, nL,
    timeLimits, nT)

  val (deltaL, deltaT) = (
    (lShellLimits._2 - lShellLimits._1)/nL,
    (timeLimits._2-timeLimits._1)/nT)

  val scalesL = Seq.fill(lSeq.length)(deltaL*mult)

  val scalesT = Seq.fill(tSeq.length)(deltaT*mult)

  val (basisL, basisT) = (
    RadialBasis.gaussianBasis(lSeq, scalesL, bias = false),
    RadialBasis.gaussianBasis(tSeq, scalesT, bias = false))

  val productBasis = basisL*basisT

  val tupleListEnc = Encoder(
    (t: (Int, Int)) => List(t._1, t._2),
    (l: List[Int]) => (l.head, l.last)
  )

  private val centers = combine(Seq(lSeq, tSeq)).map(s => (s.head, s.last))

  def _centers = centers

  val dimension = lSeq.length*tSeq.length

  val dimensionL = lSeq.length

  val dimensionT = tSeq.length

  val indexEncoder = tupleListEnc > TupleIntegerEncoder(List(lSeq.length, tSeq.length))

  override protected val f: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => productBasis(x)

}
