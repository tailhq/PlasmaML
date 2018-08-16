package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.analysis.{ChebyshevBasisGenerator, HermiteBasisGenerator, LegendreBasisGenerator, RadialBasis}
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
  phiL: Basis[Double], phiT: Basis[Double],
  phiL_l: Basis[Double], phiL_ll: Basis[Double],
  phiT_t: Basis[Double]) extends PSDBasis {


  val dimension_l: Int

  val dimension_t: Int

  override protected val f: ((Double, Double)) => DenseVector[Double] =
    (x: (Double, Double)) => (phiL(x._1)*phiT(x._2).t).toDenseVector


  override val f_l: ((Double, Double)) => DenseVector[Double] = (phiL_l*phiT).run _

  override val f_ll: ((Double, Double)) => DenseVector[Double] = (phiL_ll*phiT).run _

  override val f_t: ((Double, Double)) => DenseVector[Double] = (phiL*phiT_t).run _

}

object HybridPSDBasis {

  /**
    * Return a [[HybridPSDBasis]] consisting
    * of a Chebyshev basis in the spatial domain
    * and an Inverse Multi-Quadric basis in the
    * temporal domain.
    * */
  def chebyshev_imq_basis(
    beta_t: Double,
    lShellLimits: (Double, Double), nL: Int,
    timeLimits: (Double, Double), nT: Int,
    logScale: Boolean = false,
    biasFlag: Boolean = false): HybridPSDBasis = {

    val (_, tSeq) = RadialDiffusion.buildStencil(
      lShellLimits, nL,
      timeLimits, nT,
      (false, logScale))

    val deltaT: Double =
      if(logScale) math.log(timeLimits._2 - timeLimits._1)/nT
      else (timeLimits._2 - timeLimits._1)/nT

    val scalesT: Seq[Double] =
      if(logScale) Seq.tabulate(tSeq.length)(i =>
        if(i == 0) math.exp(deltaT)
        else if(i < nL) math.exp((i+1)*deltaT) - math.exp(i*deltaT)
        else math.exp((nL+1)*deltaT) - math.exp(nL*deltaT))
      else Seq.fill(tSeq.length)(deltaT)

    val basis_time = RadialBasis.invMultiquadricBasis(beta_t)(tSeq, scalesT, bias = biasFlag)

    val l_adj = DataPipe((l_shell: Double) => (l_shell - lShellLimits._1)/(lShellLimits._2 - lShellLimits._1))

    val l_domain_adjust = DataPipe((l: Double) => 2*l - 1d)

    val basis_space =
      (l_adj > l_domain_adjust) %>
        ChebyshevBasisGenerator(nL, 1) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          (v: DenseVector[Double]) => if(biasFlag) v else v.slice(1, v.length)
        )

    def T(n: Int, x: Double) = utils.chebyshev(n, x, kind = 1)

    def U(n: Int, x: Double) = utils.chebyshev(n, x, kind = 2)

    val basis_time_t = if (biasFlag) {
      Basis((t: Double) =>
        DenseVector(Array(0d) ++ tSeq.zip(scalesT).toArray.map(c => -beta_t*math.abs(t-c._1)/c._2)) *:* basis_time(t)
      )
    } else {
      Basis((t: Double) =>
        DenseVector(tSeq.zip(scalesT).toArray.map(c => -beta_t*math.abs(t-c._1)/c._2)) *:* basis_time(t)
      )
    }

    val basis_space_l = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector(Array(0d) ++ (1 to nL).toArray.map(i => i*U(i - 1, l))))
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector((1 to nL).toArray.map(i => i*U(i - 1, l))))
    }

    val basis_space_ll = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector(Array(0d) ++ Array.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d))
        )
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d)
        )
    }

    new HybridPSDBasis(basis_space, basis_time, basis_space_l, basis_space_ll, basis_time_t) {

      override val dimension_l: Int = if(biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 2 else nT + 1

      override val dimension: Int   = dimension_l*dimension_t
    }
  }

  def chebyshev_space_time_basis(
    lShellLimits: (Double, Double), nL: Int,
    timeLimits: (Double, Double), nT: Int,
    logScale: Boolean = false,
    biasFlag: Boolean = false): HybridPSDBasis = {

    val l_adj = DataPipe((l_shell: Double) => (l_shell - lShellLimits._1)/(lShellLimits._2 - lShellLimits._1))

    val l_domain_adjust = DataPipe((l: Double) => 2*l - 1d)

    val t_adj = DataPipe((time: Double) => (time - timeLimits._1)/(timeLimits._2 - timeLimits._1))

    val t_domain_adjust = DataPipe((t: Double) => 2*t - 1d)

    val basis_space =
      (l_adj > l_domain_adjust) %>
        ChebyshevBasisGenerator(nL, 1) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if(biasFlag) v else v.slice(1, v.length)
        )

    val basis_time =
      (t_adj > t_domain_adjust) %>
        ChebyshevBasisGenerator(nT, 1) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if(biasFlag) v else v.slice(1, v.length)
        )

    def T(n: Int, x: Double) = utils.chebyshev(n, x, kind = 1)

    def U(n: Int, x: Double) = utils.chebyshev(n, x, kind = 2)

    val basis_time_t = if(biasFlag) {
      (t_adj > t_domain_adjust) %>
        Basis((l: Double) => DenseVector(Array(0d) ++ (1 to nT).toArray.map(i => i*U(i - 1, l))))
    } else {
      (t_adj > t_domain_adjust) %>
        Basis((l: Double) => DenseVector((1 to nT).toArray.map(i => i*U(i - 1, l))))
    }

    val basis_space_l = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector(Array(0d) ++ (1 to nL).toArray.map(i => i*U(i - 1, l))))
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector((1 to nL).toArray.map(i => i*U(i - 1, l))))
    }

    val basis_space_ll = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector(Array(0d) ++ Array.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d))
        )
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d)
        )
    }

    new HybridPSDBasis(basis_space, basis_time, basis_space_l, basis_space_ll, basis_time_t) {

      override val dimension_l: Int = if(biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 1 else nT

      override val dimension: Int   = dimension_l*dimension_t
    }
  }


  def chebyshev_hermite_basis(
    lShellLimits: (Double, Double), nL: Int,
    timeLimits: (Double, Double), nT: Int,
    logScale: Boolean = false,
    biasFlag: Boolean = false): HybridPSDBasis = {

    val l_adj = DataPipe((l_shell: Double) => (l_shell - lShellLimits._1)/(lShellLimits._2 - lShellLimits._1))

    val l_domain_adjust = DataPipe((l: Double) => 2*l - 1d)

    val basis_space =
      (l_adj > l_domain_adjust) %>
        ChebyshevBasisGenerator(nL, 1) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if(biasFlag) v else v.slice(1, v.length)
        )

    val basis_time =
        HermiteBasisGenerator(nT) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if(biasFlag) v else v.slice(1, v.length)
        )

    def T(n: Int, x: Double) = utils.chebyshev(n, x, kind = 1)

    def U(n: Int, x: Double) = utils.chebyshev(n, x, kind = 2)

    val basis_time_t = if(biasFlag) {
        Basis((l: Double) => DenseVector(Array(0d) ++ (1 to nT).toArray.map(i => i*utils.hermite(i - 1, l))))
    } else {
        Basis((l: Double) => DenseVector((1 to nT).toArray.map(i => i*utils.hermite(i - 1, l))))
    }

    val basis_space_l = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector(Array(0d) ++ (1 to nL).toArray.map(i => i*U(i - 1, l))))
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector((1 to nL).toArray.map(i => i*U(i - 1, l))))
    }

    val basis_space_ll = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector(Array(0d) ++ Array.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d))
        )
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d)
        )
    }

    new HybridPSDBasis(basis_space, basis_time, basis_space_l, basis_space_ll, basis_time_t) {

      override val dimension_l: Int = if(biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 1 else nT

      override val dimension: Int   = dimension_l*dimension_t
    }
  }


  def chebyshev_laguerre_basis(
    lShellLimits: (Double, Double), nL: Int,
    timeLimits: (Double, Double), nT: Int,
    alpha: Double     = 0d,
    biasFlag: Boolean = false): HybridPSDBasis = {

    val l_adj = DataPipe((l_shell: Double) => (l_shell - lShellLimits._1)/(lShellLimits._2 - lShellLimits._1))

    val l_domain_adjust = DataPipe((l: Double) => 2*l - 1d)

    val basis_space =
      (l_adj > l_domain_adjust) %>
        ChebyshevBasisGenerator(nL, 1) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if(biasFlag) v else v.slice(1, v.length)
        )

    val basis_time =
      MagParamBasis.laguerre_basis(nT, alpha) >
        DataPipe[DenseVector[Double], DenseVector[Double]](
          v => if(biasFlag) v else v.slice(1, v.length)
        )

    def T(n: Int, x: Double) = utils.chebyshev(n, x, kind = 1)

    def U(n: Int, x: Double) = utils.chebyshev(n, x, kind = 2)

    val basis_time_t = if(biasFlag) {
      Basis((l: Double) => MagParamBasis.laguerre_basis(nT, alpha).J(l))
    } else {
      Basis((l: Double) => MagParamBasis.laguerre_basis(nT, alpha).J(l).slice(1, nT + 1))
    }

    val basis_space_l = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector(Array(0d) ++ (1 to nL).toArray.map(i => i*U(i - 1, l))))
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) => DenseVector((1 to nL).toArray.map(i => i*U(i - 1, l))))
    }

    val basis_space_ll = if(biasFlag) {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector(Array(0d) ++ Array.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d))
        )
    } else {
      (l_adj > l_domain_adjust) %>
        Basis((l: Double) =>
          DenseVector.tabulate(nL)(i =>
            if(i+1 > 1) (i+1)*((i+1)*T(i+1, l) - l*U(i, l))/(l*l - 1)
            else 0d)
        )
    }

    new HybridPSDBasis(basis_space, basis_time, basis_space_l, basis_space_ll, basis_time_t) {

      override val dimension_l: Int = if(biasFlag) nL + 1 else nL

      override val dimension_t: Int = if (biasFlag) nT + 1 else nT

      override val dimension: Int   = dimension_l*dimension_t
    }
  }


}

