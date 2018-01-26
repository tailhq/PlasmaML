package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.combine


/**
  * <h3>Phase Space Density: Basis Expansions</h3>
  * A basis function expansion (&phi;) of the plasma
  * Phase Space Density (&fnof;) in the radial diffusion
  * system.
  *
  * &part;&fnof;/&part;t =
  * L<sup>2</sup>&part;/&part;L(D<sub>LL</sub> &times; L<sup>-2</sup> &times; &part;&fnof;/&part;L)
  * - &lambda;(L,t) &times; &fnof;(L,t)
  * + Q(L,t)
  *
  * */
abstract class PSDBasis extends Basis[(Double, Double)] {

  self =>

  /**
    * Dimensionality of &phi;
    * */
  val dimension: Int

  /**
    * Calculates &part;&phi;/&part;L
    * */
  val f_l: ((Double, Double)) => DenseVector[Double]

  /**
    * Calculates &part;<sup>2</sup>&phi;/&part;L<sup>2</sup>
    * */
  val f_ll: ((Double, Double)) => DenseVector[Double]

  /**
    * Calculates &part;&phi;/&part;t
    * */
  val f_t: ((Double, Double)) => DenseVector[Double]

  /**
    * Calculates the basis &psi; resulting from applying
    * the differential operator of plasma radial diffusion,
    * on the basis set &phi;
    *
    * D = (d/dt - L<sup>2</sup>d/dL(D<sub>LL</sub> &times; L<sup>-2</sup> &times;  d/dL) + &lambda;(L,t))
    *
    * &psi;(L,t) = D[&phi;(L,t)]
    *
    * @param diffusionField The diffusion field/coefficient.
    * @param diffusionFieldGradL The first partial spatial derivative of
    *                            the diffusion field &part;D<sub>LL</sub>/&part;L
    * @param lossTimeScale The loss rate &lambda;(L,t)
    *
    * */
  def operator_basis(
    diffusionField: DataPipe[(Double, Double), Double],
    diffusionFieldGradL: DataPipe[(Double, Double), Double],
    lossTimeScale: DataPipe[(Double, Double), Double]): Basis[(Double, Double)] =
    Basis((x: (Double, Double)) => {

      val dll = diffusionField(x)
      val alpha = diffusionFieldGradL(x) - 2d*diffusionField(x)/x._1
      val lambda = lossTimeScale(x)

       f_t(x) + lambda*f(x) - (dll*f_ll(x) + alpha*f_l(x))
    })

  /**
    * Returns a [[PSDBasis]] that is the addition
    * of the current basis and the one accepted as
    * the method argument.
    *
    * &phi;(L,t) = &phi;<sub>1</sub>(L,t) + &phi;<sub>2</sub>(L,t)
    * */
  def +(other: PSDBasis): PSDBasis =
    new PSDBasis {

      override val dimension: Int = self.dimension

      override protected val f = (x: (Double, Double)) => self(x) + other(x)

      override val f_l:  ((Double, Double)) => DenseVector[Double] =
        (x: (Double, Double)) => self.f_l(x) + other.f_l(x)

      override val f_ll: ((Double, Double)) => DenseVector[Double] =
        (x: (Double, Double)) => self.f_ll(x) + other.f_ll(x)

      override val f_t:  ((Double, Double)) => DenseVector[Double] =
        (x: (Double, Double)) => self.f_t(x) + other.f_t(x)

    }

  /**
    * Returns a [[PSDBasis]] that is the concatenation
    * of the current basis and the one accepted as
    * the method argument.
    *
    * &phi;(L,t) = (&phi;<sub>1</sub>(L,t), &phi;<sub>2</sub>(L,t))
    * */
  def ::(other: PSDBasis): PSDBasis =
    new PSDBasis {

      override val dimension: Int = self.dimension + other.dimension

      override protected val f = (x: (Double, Double)) => DenseVector.vertcat(self(x), other(x))

      override val f_l:  ((Double, Double)) => DenseVector[Double] =
        (x: (Double, Double)) => DenseVector.vertcat(self.f_l(x), other.f_l(x))

      override val f_ll: ((Double, Double)) => DenseVector[Double] =
        (x: (Double, Double)) => DenseVector.vertcat(self.f_ll(x), other.f_ll(x))

      override val f_t:  ((Double, Double)) => DenseVector[Double] =
        (x: (Double, Double)) => DenseVector.vertcat(self.f_t(x), other.f_t(x))

    }

}

/**
  * <h3>Radial Basis Phase Space Density Expansions</h3>
  *
  * A top level class that can be extended to implement
  * various radial basis function based PSD expansions.
  *
  * Nodes are placed on a regular or logarithmically
  * spaced space-time grid.
  *
  * @param lShellLimits Spatial limits
  * @param nL Number of spatial intervals
  * @param timeLimits Temporal limits
  * @param nT Number of temporal intervals
  * @param logScaleFlags Set to true for generating
  *                      logarithmically spaced grid,
  *                      one flag each for space and
  *                      time.
  * */
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

  def scalesL: Seq[Double] =
    if(logScaleFlags._1) Seq.tabulate(lSeq.length)(i =>
      if(i == 0) math.exp(deltaL)
      else if(i < nL) math.exp((i+1)*deltaL) - math.exp(i*deltaL)
      else math.exp((nL+1)*deltaL) - math.exp(nL*deltaL)).map(_*mult)
    else Seq.fill(lSeq.length)(deltaL*mult)

  def scalesT: Seq[Double] =
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

  protected def scales: Seq[(Double, Double)] = combine(Seq(scalesL, scalesT)).map(s => (s.head, s.last))

  def _centers: Seq[(Double, Double)] = centers

  override val dimension: Int = lSeq.length*tSeq.length

  val dimensionL: Int = lSeq.length

  val dimensionT: Int = tSeq.length

  val indexEncoder: Encoder[(Int, Int), Int] = tupleListEnc > TupleIntegerEncoder(List(lSeq.length, tSeq.length))

}