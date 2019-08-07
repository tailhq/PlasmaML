package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.analysis.{
  ChebyshevBasisGenerator,
  HermiteBasisGenerator,
  LegendreBasisGenerator,
  RadialBasis
}
import io.github.mandar2812.dynaml.pipes.{Basis, DataPipe}
import io.github.mandar2812.dynaml.utils

/**
  * <h3>Phase Space Density: Hybrid Basis Expansions</h3>
  *
  * A general basis expansion for the Phase Space Density.
  *
  * Consists of an outer product separate basis expansions
  * for space and time.
  *
  * */
abstract class HybridPSDBasis(
  phiL: Basis[Double],
  phiT: Basis[Double],
  phiL_l: Basis[Double],
  phiL_ll: Basis[Double],
  phiT_t: Basis[Double])
    extends PSDBasis {

  val dimension_l: Int

  val dimension_t: Int

  override protected val f: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => (phiL(x._1) * phiT(x._2).t).toDenseVector

  override val f_l: ((Double, Double)) => DenseVector[Double] =
    (phiL_l * phiT).run _

  override val f_ll: ((Double, Double)) => DenseVector[Double] =
    (phiL_ll * phiT).run _

  override val f_t: ((Double, Double)) => DenseVector[Double] =
    (phiL * phiT_t).run _

}

object HybridPSDBasis {

  /**
    * Creates a Chebyshev basis function expansion.
    *
    * [C<sub>1</sub>(x), ... , C<sup>n</sup>(x)]
    *
    * @param domainLimits The lower and upper limits of the input domain X.
    * @param maxDegree The maximum degree polynomial up to which the basis is constructed.
    * @param biasFlag Set to true, if C<sub>0</sub>(x) should also be included in the basis,
    *                 defaults to false.
    * @param kind Choose the Chebyshev polynomial of the first or second kind, defaults to 1.
    */
  def chebyshev_basis(
    domainLimits: (Double, Double),
    maxDegree: Int,
    biasFlag: Boolean = false,
    kind: Int = 1
  ): (Basis[Double], Basis[Double], Basis[Double]) = {

    def T(n: Int, x: Double) = utils.chebyshev(n, x, kind = 1)

    def U(n: Int, x: Double) = utils.chebyshev(n, x, kind = 2)

    def dT(n: Int, x: Double) = n * U(n - 1, x)

    def dU(n: Int, x: Double) =
      ((n + 1) * T(n + 1, x) - x * U(n, x)) / (x * x - 1d)

    def d2T(n: Int, x: Double) = n * dU(n - 1, x)

    def d2U(n: Int, x: Double) = {
      val denominator = x * x - 1d
      val numerator   = (n + 1) * T(n + 1, x) - x * U(n, x)

      ((denominator * ((n + 1) * dT(n + 1, x) - U(n, x) - x * dU(n, x))) - (numerator * 2 * x)) / (denominator * denominator)
    }

    val deltaL = domainLimits._2 - domainLimits._1

    val grad_scaling_factor = 2d / deltaL

    val l_adj = DataPipe(
      (l_shell: Double) => (l_shell - domainLimits._1) / deltaL
    )

    val l_domain_adjust = DataPipe((l: Double) => 2 * l - 1d)

    val basis_space =
      (l_adj > l_domain_adjust) %>
        ChebyshevBasisGenerator(maxDegree, kind) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if (biasFlag) v else v.slice(1, v.length)
        )

    def df_dl(n: Int, x: Double) =
      (if (kind == 1) dT(n, x) else dU(n, x)) * grad_scaling_factor

    def d2f_dl2(n: Int, x: Double) =
      (if (kind == 1) d2T(n, x) else d2U(n, x)) * grad_scaling_factor * grad_scaling_factor

    val basis_space_l =
      (l_adj > l_domain_adjust) %>
        Basis((z: Double) => {
          val us =
            (1 to maxDegree).toArray.map(i => df_dl(i, z))

          DenseVector(
            if (biasFlag) Array(0d) ++ us
            else us
          )
        })

    val basis_space_ll =
      (l_adj > l_domain_adjust) %>
        Basis((z: Double) => {
          val us = Array
            .tabulate(maxDegree)(
              i =>
                if (i + 1 > 1) d2f_dl2(i, z)
                else 0d
            )

          DenseVector(
            if (biasFlag) Array(0d) ++ us
            else us
          )
        })

    (basis_space, basis_space_l, basis_space_ll)

  }

  /**
    * Return a [[HybridPSDBasis]] consisting
    * of a Chebyshev basis in the spatial domain
    * and an Inverse Multi-Quadric basis in the
    * temporal domain.
    * */
  def chebyshev_imq_basis(
    beta_t: Double,
    lShellLimits: (Double, Double),
    nL: Int,
    timeLimits: (Double, Double),
    nT: Int,
    logScale: Boolean = false,
    biasFlag: Boolean = false,
    kind: Int = 1
  ): HybridPSDBasis = {

    val (_, tSeq) = RadialDiffusion.buildStencil(
      lShellLimits,
      nL,
      timeLimits,
      nT,
      (false, logScale)
    )

    val (basis_space, basis_space_l, basis_space_ll) =
      chebyshev_basis(lShellLimits, nL, biasFlag, kind)

    val deltaT: Double =
      if (logScale) math.log(timeLimits._2 - timeLimits._1) / nT
      else (timeLimits._2 - timeLimits._1) / nT

    val scalesT: Seq[Double] =
      if (logScale)
        Seq.tabulate(tSeq.length)(
          i =>
            if (i == 0) math.exp(deltaT)
            else if (i < nL) math.exp((i + 1) * deltaT) - math.exp(i * deltaT)
            else math.exp((nL + 1) * deltaT) - math.exp(nL * deltaT)
        )
      else Seq.fill(tSeq.length)(deltaT)

    val basis_time =
      RadialBasis.invMultiquadricBasis(beta_t)(tSeq, scalesT, bias = biasFlag)

    val basis_time_t = if (biasFlag) {
      Basis(
        (t: Double) =>
          DenseVector(
            Array(0d) ++ tSeq
              .zip(scalesT)
              .toArray
              .map(c => -beta_t * math.abs(t - c._1) / c._2)
          ) *:* basis_time(t)
      )
    } else {
      Basis(
        (t: Double) =>
          DenseVector(
            tSeq
              .zip(scalesT)
              .toArray
              .map(c => -beta_t * math.abs(t - c._1) / c._2)
          ) *:* basis_time(t)
      )
    }

    new HybridPSDBasis(
      basis_space,
      basis_time,
      basis_space_l,
      basis_space_ll,
      basis_time_t
    ) {

      override val dimension_l: Int = if (biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 2 else nT + 1

      override val dimension: Int = dimension_l * dimension_t
    }
  }

  def chebyshev_space_time_basis(
    lShellLimits: (Double, Double),
    nL: Int,
    timeLimits: (Double, Double),
    nT: Int,
    logScale: Boolean = false,
    biasFlag: Boolean = false,
    kind: (Int, Int) = (1, 1)
  ): HybridPSDBasis = {

    val (basis_space, basis_space_l, basis_space_ll) =
      chebyshev_basis(lShellLimits, nL, biasFlag, kind._1)

    val (basis_time, basis_time_t, _) =
      chebyshev_basis(timeLimits, nT, biasFlag, kind._2)

    new HybridPSDBasis(
      basis_space,
      basis_time,
      basis_space_l,
      basis_space_ll,
      basis_time_t
    ) {

      override val dimension_l: Int = if (biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 1 else nT

      override val dimension: Int = dimension_l * dimension_t
    }
  }

  def chebyshev_hermite_basis(
    lShellLimits: (Double, Double),
    nL: Int,
    timeLimits: (Double, Double),
    nT: Int,
    logScale: Boolean = false,
    biasFlag: Boolean = false,
    kind: Int = 1
  ): HybridPSDBasis = {

    val (basis_space, basis_space_l, basis_space_ll) =
      chebyshev_basis(lShellLimits, nL, biasFlag, kind)

    val basis_time =
      HermiteBasisGenerator(nT) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if (biasFlag) v else v.slice(1, v.length)
        )

    val basis_time_t = if (biasFlag) {
      Basis(
        (l: Double) =>
          DenseVector(
            Array(0d) ++ (1 to nT).toArray.map(i => i * utils.hermite(i - 1, l))
          )
      )
    } else {
      Basis(
        (l: Double) =>
          DenseVector((1 to nT).toArray.map(i => i * utils.hermite(i - 1, l)))
      )
    }

    new HybridPSDBasis(
      basis_space,
      basis_time,
      basis_space_l,
      basis_space_ll,
      basis_time_t
    ) {

      override val dimension_l: Int = if (biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 1 else nT

      override val dimension: Int = dimension_l * dimension_t
    }
  }

  def chebyshev_laguerre_basis(
    lShellLimits: (Double, Double),
    nL: Int,
    timeLimits: (Double, Double),
    nT: Int,
    alpha: Double = 0d,
    biasFlag: Boolean = false,
    kind: Int = 1
  ): HybridPSDBasis = {

    val (basis_space, basis_space_l, basis_space_ll) =
      chebyshev_basis(lShellLimits, nL, biasFlag, kind)

    val basis_time =
      MagParamBasis.laguerre_basis(nT, alpha) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if (biasFlag) v else v.slice(1, v.length)
        )

    val basis_time_t = if (biasFlag) {
      Basis((l: Double) => MagParamBasis.laguerre_basis(nT, alpha).J(l))
    } else {
      Basis(
        (l: Double) =>
          MagParamBasis.laguerre_basis(nT, alpha).J(l).slice(1, nT + 1)
      )
    }
    new HybridPSDBasis(
      basis_space,
      basis_time,
      basis_space_l,
      basis_space_ll,
      basis_time_t
    ) {

      override val dimension_l: Int = if (biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 1 else nT

      override val dimension: Int = dimension_l * dimension_t
    }
  }

}
