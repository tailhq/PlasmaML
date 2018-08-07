import breeze.stats.distributions._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.probability.mcmc._
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel
import io.github.mandar2812.PlasmaML.dynamics.diffusion.BasisFuncRadialDiffusionModel
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._
import io.github.mandar2812.dynaml.repl.Router.main

@main
def main(
  nData: Int = 20, nBoundary:Int = 20,
  nColData: Int = 40) = {

  num_bulk_data = nData
  num_boundary_data = nBoundary

  num_dummy_data = nColData

  lambda_params = (-1, 3.15, 0d, -0.2)

  q_params = (0d, 2.5d, 0.05, 0.45)

  nL = 300
  nT = 200

  initialPSD = (l: Double) => {
    val c = utils.chebyshev(3, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
    4000d + 1000*c - 1000*utils.chebyshev(5, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
  }

  val rds = RDExperiment.solver(lShellLimits, timeLimits, nL, nT)


  val basisSize = (9, 9)

  val hybrid_basis = new InverseMQPSDBasis(1d)(
    lShellLimits, 14, timeLimits, 19, (false, false)
  )

  hybrid_basis.mult = 0.5

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_imq_basis(
    0.75, lShellLimits, basisSize._1,
    timeLimits, basisSize._2, biasFlag = true)

  val burn = 1500

  val seKernel = new GenExpSpaceTimeKernel[Double](
    1d, deltaL, deltaT)(
    sqNormDouble, l1NormDouble)

  val noiseKernel = new DiracTuple2Kernel(1.5)

  noiseKernel.block_all_hyper_parameters

  val (solution, (boundary_data, bulk_data), colocation_points) = RDExperiment.generateData(
    rds, dll, lambda, Q, initialPSD)(
    measurement_noise, num_boundary_data,
    num_bulk_data, num_dummy_data)

  val model = new BasisFuncRadialDiffusionModel(
    Kp, dll_params, (0d, 0.2, 0d, 0.2), q_params)(
    seKernel, noiseKernel,
    boundary_data ++ bulk_data, colocation_points,
    chebyshev_hybrid_basis::hybrid_basis, dualFlag = false
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(c =>
        c.contains("dll") ||
        c.contains("base::") ||
        c.contains("lambda_gamma") ||
        c.contains("Q_")
      )
  }


  model.block(blocked_hyp:_*)

  val eff_hyp = model.effective_hyper_parameters
  val hyper_prior = {
    eff_hyp.filter(_.contains("base::")).map(h => (h, new LogNormal(0d, 2d))).toMap ++
      eff_hyp.filterNot(h => h.contains("base::") || h.contains("tau")).map(h => (h, new Gaussian(0d, 2.5d))).toMap ++
      Map(
        "lambda_alpha" -> new Gaussian(0d, 2d),
        "lambda_beta" -> new LogNormal(0d, 2d),
        "lambda_b" -> new Gaussian(0d, 2.0)).filterKeys(eff_hyp.contains)
  }

  model.regCol = 0.05
  model.regObs = 1d
  model.reg = 0d

  //Create the MCMC sampler
  val mcmc_sampler = new AdaptiveHyperParameterMCMC[
    model.type, ContinuousDistr[Double]](
    model, hyper_prior, burn)

  val num_post_samples = 1000

  //Draw samples from the posterior
  val samples = mcmc_sampler.iid(num_post_samples).draw

  val resPath = RDExperiment.writeResults(
    solution, boundary_data, bulk_data, colocation_points,
    hyper_prior, samples, basisSize, "HybridMQ",
    (model.regCol, model.regObs))

  val scriptPath = pwd / "mag-core" / 'scripts / "visualiseSamplingResults.R"

  %%('Rscript, scriptPath.toString, resPath.toString, "loss")


  RDExperiment.samplingReport(
    samples, eff_hyp.map(c => (c, quantities_loss(c))).toMap,
    gt, mcmc_sampler.sampleAcceptenceRate)

  RDExperiment.visualisePSD(lShellLimits, timeLimits, nL, nT)(initialPSD, solution, Kp)

  RDExperiment.visualiseResultsLoss(samples, gt, hyper_prior)

  (samples, model, solution, resPath)
}
