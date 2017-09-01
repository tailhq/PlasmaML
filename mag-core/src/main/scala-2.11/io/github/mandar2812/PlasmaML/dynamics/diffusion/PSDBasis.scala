package io.github.mandar2812.PlasmaML.dynamics.diffusion

import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.combine

/**
  * A set of characteristics which represent
  * a basis function expansion of the plasma
  * Phase Space Density in the radial diffusion
  * system.
  *
  * df/dt = L<sup>2</sup>d/dL(D<sub>LL</sub> &times; L<sup>-2</sup> &times;  df/dL)
  * - &lambda;(L,t) &times; f(L,t)
  *
  * */
trait PSDBasis extends Basis[(Double, Double)]{

  val dimension: Int

  val dimensionL: Int

  val dimensionT: Int

  /**
    * Calculate the function which must be multiplied to the current
    * basis in order to obtain the operator transformed basis.
    * */
  def operator_basis(
    diffusionField: DataPipe[(Double, Double), Double],
    diffusionFieldGradL: DataPipe[(Double, Double), Double],
    lossTimeScale: DataPipe[(Double, Double), Double]): Basis[(Double, Double)]

}


abstract class PSDRadialBasis(
  val lShellLimits: (Double, Double), val nL: Int,
  val timeLimits: (Double, Double), val nT: Int,
  val logScaleFlags: (Boolean, Boolean) = (false, false))
  extends PSDBasis {

  var mult = 1d

  val (lSeq, tSeq) = RadialDiffusion.buildStencil(
    lShellLimits, nL,
    timeLimits, nT,
    logScaleFlags)

  val deltaL: Double =
    if(logScaleFlags._1) math.log(lShellLimits._2 - lShellLimits._1)/nL
    else (lShellLimits._2 - lShellLimits._1)/nL

  val deltaT: Double =
    if(logScaleFlags._2) math.log(timeLimits._2 - timeLimits._1)/nT
    else (timeLimits._2 - timeLimits._1)/nT

  val scalesL: Seq[Double] =
    if(logScaleFlags._1) Seq.tabulate(lSeq.length)(i =>
      if(i == 0) math.exp(deltaL)
      else if(i < nL) math.exp((i+1)*deltaL) - math.exp(i*deltaL)
      else math.exp((nL+1)*deltaL) - math.exp(nL*deltaL)).map(_*mult)
    else Seq.fill(lSeq.length)(deltaL*mult)

  val scalesT: Seq[Double] =
    if(logScaleFlags._2) Seq.tabulate(tSeq.length)(i =>
      if(i == 0) math.exp(deltaT)
      else if(i < nL) math.exp((i+1)*deltaT) - math.exp(i*deltaT)
      else math.exp((nL+1)*deltaT) - math.exp(nL*deltaT)).map(_*mult)
    else Seq.fill(tSeq.length)(deltaT*mult)

  val tupleListEnc = Encoder(
    (t: (Int, Int)) => List(t._1, t._2),
    (l: List[Int]) => (l.head, l.last)
  )

  protected val centers: Seq[(Double, Double)] = combine(Seq(lSeq, tSeq)).map(s => (s.head, s.last))

  protected val scales: Seq[(Double, Double)] = combine(Seq(scalesL, scalesT)).map(s => (s.head, s.last))

  def _centers: Seq[(Double, Double)] = centers

  override val dimension: Int = lSeq.length*tSeq.length

  override val dimensionL: Int = lSeq.length

  override val dimensionT: Int = tSeq.length

  val indexEncoder: Encoder[(Int, Int), Int] = tupleListEnc > TupleIntegerEncoder(List(lSeq.length, tSeq.length))



}