package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder}
import io.github.mandar2812.dynaml.probability.{ContinuousDistrRV, MatrixNormalRV}
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
  *                       in the spatial domain, i.e. the third invariant of plasma motion, L-shell.
  * @param psdCovarianceT Covariance of the PSD in the temporal domain.
  * @param injectionProcess A gaussian process prior on the injection function Q(l,t)
  * @param diffusionProcess A gaussian process prior on the diffusion field D(l,t)
  * @author mandar2812 date 04/05/2017.
  * */
class StochasticRadialDiffusion[ParamsQ, ParamsD, ParamsL](
  psdCovarianceL: StochasticRadialDiffusion.Kernel,
  psdCovarianceT: StochasticRadialDiffusion.Kernel,
  val injectionProcess: StochasticRadialDiffusion.LatentProcess[ParamsQ],
  val diffusionProcess: StochasticRadialDiffusion.LatentProcess[ParamsD],
  val lossProcess: StochasticRadialDiffusion.LatentProcess[ParamsL]) extends Serializable {

  private val logger = Logger.getLogger(this.getClass)

  type DomainLimits = (Double, Double)
  type TimeSlice = DenseVector[Double]

  var num_samples: Int = 10000

  var ensembleMode: Boolean = false

  val (qEnc, dEnc, lEnc) = (
    injectionProcess.trendParamsEncoder,
    diffusionProcess.trendParamsEncoder,
    lossProcess.trendParamsEncoder)

  protected var state: Map[String, Double] =
    qEnc(injectionProcess._meanFuncParams) ++
      dEnc(diffusionProcess._meanFuncParams) ++
      lEnc(lossProcess._meanFuncParams) ++
      psdCovarianceL.state ++
      psdCovarianceT.state ++
      injectionProcess.covariance.state ++
      diffusionProcess.covariance.state ++
      lossProcess.covariance.state

  protected val hyper_parameters: List[String] = state.keys.toList

  injectionProcess.covariance.block_all_hyper_parameters
  diffusionProcess.covariance.block_all_hyper_parameters
  lossProcess.covariance.block_all_hyper_parameters
  psdCovarianceL.block_all_hyper_parameters
  psdCovarianceT.block_all_hyper_parameters

  protected var blocked_hyper_parameters: List[String] =
    injectionProcess.covariance.hyper_parameters ++
      diffusionProcess.covariance.hyper_parameters ++
      lossProcess.covariance.hyper_parameters ++
      psdCovarianceL.hyper_parameters ++
      psdCovarianceT.hyper_parameters

  def block(h: String*) = blocked_hyper_parameters = List(h:_*)

  def block_++(h: String*) = blocked_hyper_parameters ++= List(h:_*)

  def _state = state

  def effective_state:Map[String, Double] =
    state.filterNot(h => blocked_hyper_parameters.contains(h._1))

  def effective_hyper_parameters: List[String] =
    hyper_parameters.filterNot(h => blocked_hyper_parameters.contains(h))

  def setState(s: Map[String, Double]) = {
    assert(effective_hyper_parameters.forall(s.contains),
      "All hyper parameters must be contained in the arguments")
    effective_hyper_parameters.foreach((key) => {
      state += (key -> s(key))
    })

    injectionProcess.meanFuncParams_(qEnc.i(state))
    injectionProcess.covariance.setHyperParameters(effective_state)

    diffusionProcess.meanFuncParams_(dEnc.i(state))
    diffusionProcess.covariance.setHyperParameters(effective_state)

    lossProcess.meanFuncParams_(lEnc.i(state))
    lossProcess.covariance.setHyperParameters(effective_state)

    psdCovarianceL.setHyperParameters(effective_state)
    psdCovarianceT.setHyperParameters(effective_state)
  }

  /**
    * A function which takes as input the domain
    * stencil and returns a radial diffusion solver.
    * */
  protected val forwardSolver: (DomainLimits, Int, DomainLimits, Int) => RadialDiffusion =
    (lDomain: DomainLimits, nL: Int, timeDomain: DomainLimits, nT: Int) =>
      RadialDiffusion(lDomain, timeDomain, nL, nT)

  /**
    * Return the finite dimensional prior of the
    * injection and diffusion field on the domain stencil
    * */
  def epistemics(l_values: Seq[Double], t_values: Seq[Double]): (MatrixNormalRV, MatrixNormalRV, MatrixNormalRV) =
    (
      injectionProcess.priorDistribution(l_values, t_values),
      diffusionProcess.priorDistribution(l_values, t_values),
      lossProcess.priorDistribution(l_values, t_values))

  /**
    * Return the distribution of f(L,t), conditioned on the most likely
    * values of the injection and diffusion processes, on a
    * stencil defined by the method parameters.
    *
    * @param lDomain Lower and upper limits of L-shell.
    * @param nL Number of equally spaced points in space
    * @param timeDomain Lower and upper limits of time.
    * @param nT Number of equally spaced points in time.
    * */
  def forwardModel(
    lDomain: DomainLimits, nL: Int,
    timeDomain: DomainLimits, nT: Int)(
    f0: TimeSlice): MatrixNormalRV =
    if (!ensembleMode) StochasticRadialDiffusion.likelihood(
      psdCovarianceL, psdCovarianceT,
      injectionProcess, diffusionProcess,
      lossProcess)(lDomain, nL, timeDomain, nT)(f0)
    else StochasticRadialDiffusion.marginalLikelihood(
      psdCovarianceL, psdCovarianceT,
      injectionProcess, diffusionProcess,
      lossProcess)(lDomain, nL, timeDomain, nT)(num_samples, f0)
}

object StochasticRadialDiffusion {

  private val logger = Logger.getLogger(this.getClass)

  type LatentProcess[MP] = CoRegGPPrior[Double, Double, MP]
  type Kernel = LocalScalarKernel[Double]
  type EpistemicUncertainties = (MatrixNormalRV, MatrixNormalRV, MatrixNormalRV)
  type DomainLimits = (Double, Double)
  type TimeSlice = DenseVector[Double]


  /**
    * Convenience method
    * */
  def apply[MPD, MPQ, MPL](
    psdCovarianceL: Kernel,
    psdCovarianceT: Kernel,
    injectionProcess: LatentProcess[MPQ],
    diffusionProcess: LatentProcess[MPD],
    lossProcess: LatentProcess[MPL]) =
    new StochasticRadialDiffusion(
      psdCovarianceL, psdCovarianceT,
      injectionProcess, diffusionProcess,
      lossProcess)


  /**
    * Return the finite dimensional prior of the
    * injection and diffusion field on the domain stencil
    * */
  def epistemics[MPQ, MPD, MPL](
    injectionProcess: LatentProcess[MPQ],
    diffusionProcess: LatentProcess[MPD],
    lossProcess: LatentProcess[MPL])(
    l_values: Seq[Double],
    t_values: Seq[Double]): EpistemicUncertainties =
    (
      injectionProcess.priorDistribution(l_values, t_values),
      diffusionProcess.priorDistribution(l_values, t_values),
      lossProcess.priorDistribution(l_values, t_values))


  def likelihood[MPQ, MPD, MPL](
    psdCovarianceL: Kernel, psdCovarianceT: Kernel,
    injectionProcess: LatentProcess[MPQ],
    diffusionProcess: LatentProcess[MPD],
    lossProcess: LatentProcess[MPL])(
    lDomain: DomainLimits, nL: Int,
    timeDomain: DomainLimits, nT: Int)(
    f0: TimeSlice) = {

    logger.info("Initializing radial diffusion forward solver")

    val radialSolver = RadialDiffusion(lDomain, timeDomain, nL, nT)

    val (l_values, t_values) = RadialDiffusion.buildStencil(lDomain, nL, timeDomain, nT)

    logger.info("Constructing prior distributions of injection and diffusion fields on domain stencil")
    val (q_dist, dll_dist, loss_dist) = epistemics(
      injectionProcess, diffusionProcess, lossProcess)(
      l_values, t_values)

    val dll_profile = dll_dist.underlyingDist.m
    val q_profile = q_dist.underlyingDist.m
    val loss_profile = loss_dist.underlyingDist.m

    logger.info("Running radial diffusion system forward model on domain")
    val solution = radialSolver.solve(q_profile, dll_profile, loss_profile)(f0)

    logger.info("Approximate solution obtained, constructing distribution of PSD")
    val m = DenseMatrix.horzcat(solution.tail.map(_.asDenseMatrix.t):_*)

    logger.info("Constructing covariance matrices of PSD")
    val u = psdCovarianceL.buildKernelMatrix(l_values, l_values.length).getKernelMatrix()
    val v = psdCovarianceT.buildKernelMatrix(t_values.tail, t_values.tail.length).getKernelMatrix()

    MatrixNormalRV(m, u, v)


  }


  def marginalLikelihood[MPQ, MPD, MPL](
    psdCovarianceL: Kernel, psdCovarianceT: Kernel,
    injectionProcess: LatentProcess[MPQ],
    diffusionProcess: LatentProcess[MPD],
    lossProcess: LatentProcess[MPL])(
    lDomain: DomainLimits, nL: Int,
    timeDomain: DomainLimits, nT: Int)(
    num_samples: Int,
    f0: TimeSlice) = {

    logger.info("Initializing radial diffusion forward solver")

    val radialSolver = RadialDiffusion(lDomain, timeDomain, nL, nT)

    val (l_values, t_values) = RadialDiffusion.buildStencil(lDomain, nL, timeDomain, nT)

    logger.info("Constructing prior distributions of injection and diffusion fields on domain stencil")
    val (q_dist, dll_dist, loss_dist) =
      epistemics(
        injectionProcess, diffusionProcess,
        lossProcess)(
        l_values, t_values)

    logger.info("Generating ensemble of diffusion and injection fields.")

    val avg_solution = ensembleAvg(
      q_dist, dll_dist, loss_dist,
      radialSolver, num_samples)(f0)

    logger.info("Ensemble solution obtained")

    logger.info("Constructing covariance matrices of PSD")
    val u = psdCovarianceL.buildKernelMatrix(l_values, l_values.length).getKernelMatrix()
    val v = psdCovarianceT.buildKernelMatrix(t_values.tail, t_values.tail.length).getKernelMatrix()

    MatrixNormalRV(avg_solution, u, v)
  }

  /**
    * Calculate the ensemble averaged radial diffusion solution
    * on a rectangular domain stencil.
    *
    * @param injection_dist The distribution of the injection Q(l,t) realised on the domain stencil
    * @param diffusion_dist The distribution of the diffusion coefficient D<sup>LL</sup>(l,t)
    *                       realised on the domain stencil
    * @param radialSolver An instance of [[RadialDiffusion]]
    * @param num_samples Size of the ensemble
    * @param f0 The initial phase space density realized on the domain stencil.
    *
    * */
  def ensembleAvg(
    injection_dist: MatrixNormalRV, diffusion_dist: MatrixNormalRV, loss_dist: MatrixNormalRV,
    radialSolver: RadialDiffusion, num_samples: Int)(f0: DenseVector[Double]) = {

    val avg_solution = DenseMatrix.zeros[Double](radialSolver.nL+1, radialSolver.nT)

    var l = 1
    while (l <= num_samples) {

      val solution = radialSolver.solve(
        injection_dist.draw,
        diffusion_dist.draw,
        DenseMatrix.zeros[Double](radialSolver.nL+1, radialSolver.nT+1))(f0)

      val solutionMat = DenseMatrix.horzcat(solution.tail.map(_.asDenseMatrix.t):_*)/num_samples.toDouble

      if(l%500 == 0) {
        logger.info("\tProgress: "+"%4f".format(l*100d/num_samples)+"%")
      }

      avg_solution :+= solutionMat

      l += 1
    }

    avg_solution
  }


}





