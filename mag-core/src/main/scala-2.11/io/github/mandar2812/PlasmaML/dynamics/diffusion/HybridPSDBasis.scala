package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.analysis.{ChebyshevBasisGenerator, RadialBasis}
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
    logScale: Boolean = false) = {

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

    val basis_time = RadialBasis.invMultiquadricBasis(beta_t)(tSeq, scalesT, bias = false)

    val basis_space =
      ChebyshevBasisGenerator(nL, 1) >
        DataPipe[DenseVector[Double], DenseVector[Double]]((v: DenseVector[Double]) => v.slice(1, v.length))

    def l_adj(l_shell: Double) = (l_shell - lShellLimits._1)/(lShellLimits._2 - lShellLimits._1)

    def T(n: Int, x: Double) = utils.chebyshev(n, x, kind = 1)

    def U(n: Int, x: Double) = utils.chebyshev(n, x, kind = 2)

    val basis_time_t = Basis((t: Double) =>
      DenseVector(tSeq.zip(scalesT).toArray.map(c => -beta_t*math.abs(t-c._1)/c._2)) *:* basis_time(t)
    )

    val basis_space_l = Basis((l: Double) => {
      val l_star = 2*l_adj(l) - 1
      DenseVector.tabulate(nL)(i => if(i+1 > 0) (i+1)*U(i, l_star) else 0d)
    })

    val basis_space_ll = Basis((l: Double) => {
      val l_star = 2*l_adj(l) - 1
      DenseVector.tabulate(nL)(i =>
        if(i+1 > 1) (i+1)*((i+1)*T(i+1, l_star) - l_star*U(i, l_star))/(l_star*l_star - 1)
        else 0d
      )
    })

    new HybridPSDBasis(basis_space, basis_time, basis_space_l, basis_space_ll, basis_time_t) {
      override val dimension: Int = nL*(nT+1)
    }
  }
}

