package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.probability.MatrixNormalRV
import org.apache.log4j.Logger

/**
  * A radial diffusion system where the injection and diffusion functions
  * are treated as latent i.e. not directly observed.
  *
  * The processes of injection Q(l,t) and diffusion D<sub>LL</sub>(l,t) are modeled using the
  * [[io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior]] implementations from
  * DynaML.
  *
  * df/dt = L<sup>2</sup>d/dL(D<sub>LL</sub> &times; L<sup>-2</sup> &times;  df/dL) + Q(L,t)
  *
  * Q(L,t) ~ GP(q(L,t), C<sub>ql</sub>(L,L'), C<sub>qt</sub>(t,t'))
  *
  * D<sub>LL</sub>(L,t) ~ GP(d(L,t), C<sub>dl</sub>(L,L'), C<sub>dt</sub>(t,t'))
  *
  * See [[RadialDiffusion]] for an introduction to the radial diffusion solver.
  *
  * @param psdCovarianceL Covariance of the phase space density (PSD) f(l,t)
  *                       in the spatial domain, i.e. the third invariant of plastma motion, L-shell.
  * @param psdCovarianceT Covariance of the PSD in the temporal domain.
  * @param injectionProcess A gaussian process prior on the injection function Q(l,t)
  * @param diffusionProcess A gaussian process prior on the diffusion field D(l,t)
  * @param linearDecay Same function as [[RadialDiffusion.linearDecay]]
  * @author mandar2812 date 04/05/2017.
  * */
class StochasticRadialDiffusion[ParamsQ, ParamsD](
  psdCovarianceL: StochasticRadialDiffusion.Kernel,
  psdCovarianceT: StochasticRadialDiffusion.Kernel,
  injectionProcess: StochasticRadialDiffusion.LatentProcess[ParamsQ],
  diffusionProcess: StochasticRadialDiffusion.LatentProcess[ParamsD],
  linearDecay: Boolean = false) {

  private val logger = Logger.getLogger(this.getClass)

  type DomainLimits = (Double, Double)
  type TimeSlice = DenseVector[Double]

  /**
    * A function which takes as input the domain
    * stencil and returns a radial diffusion solver.
    * */
  protected val forwardSolver: (DomainLimits, Int, DomainLimits, Int) => RadialDiffusion =
    (lDomain: DomainLimits, nL: Int, timeDomain: DomainLimits, nT: Int) =>
      RadialDiffusion(lDomain, timeDomain, nL, nT, linearDecay)

  /**
    * Return the finite dimensional prior of the
    * injection and diffusion field on the domain stencil
    * */
  def epistemics(l_values: Seq[Double], t_values: Seq[Double]): (MatrixNormalRV, MatrixNormalRV) =
    (
      injectionProcess.priorDistribution(l_values, t_values),
      diffusionProcess.priorDistribution(l_values, t_values))

  /**
    * Return the prior distribution of f(L,t) on a
    * stencil defined by the method parameters.
    *
    * @param lDomain Lower and upper limits of L-shell.
    * @param nL Number of equally spaced points in space
    * @param timeDomain Lower and upper limits of time.
    * @param nT Number of equally spaced points in time.
    * */
  def priorDistribution(
    lDomain: DomainLimits, nL: Int,
    timeDomain: DomainLimits, nT: Int)(
    f0: TimeSlice): MatrixNormalRV = {

    logger.info("Initializing radial diffusion forward solver")
    val radialSolver = forwardSolver(lDomain, nL, timeDomain, nT)

    val (l_values, t_values) = radialSolver.stencil

    logger.info("Constructing prior distributions of injection and diffusion fields on domain stencil")
    val (q_dist, dll_dist) = epistemics(l_values, t_values)

    val dll_profile = dll_dist.draw
    val q_profile = q_dist.draw

    logger.info("Running radial diffusion system forward model on domain")
    val solution = radialSolver.solve(q_profile, dll_profile, DenseMatrix.zeros[Double](nL+1, nT))(f0)

    logger.info("Approximate solution obtained, constructing distribution of PSD")
    val m = DenseMatrix.horzcat(solution.tail.map(_.asDenseMatrix.t):_*)

    logger.info("Constructing covariance matrices of PSD")
    val u = psdCovarianceL.buildKernelMatrix(l_values, l_values.length).getKernelMatrix()
    val v = psdCovarianceT.buildKernelMatrix(t_values.tail, t_values.tail.length).getKernelMatrix()

    MatrixNormalRV(m, u, v)
  }

}

object StochasticRadialDiffusion {

  type LatentProcess[MP] = CoRegGPPrior[Double, Double, MP]
  type Kernel = LocalScalarKernel[Double]

  /**
    * Convenience method
    * */
  def apply[MPD, MPQ](
    psdCovarianceL: Kernel,
    psdCovarianceT: Kernel,
    injectionProcess: LatentProcess[MPQ],
    diffusionProcess: LatentProcess[MPD],
    linearDecay: Boolean = false) =
    new StochasticRadialDiffusion(
      psdCovarianceL, psdCovarianceT,
      injectionProcess, diffusionProcess)

}