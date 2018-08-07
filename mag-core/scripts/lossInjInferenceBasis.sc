import io.github.mandar2812.dynaml.repl.Router.main
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.probability.mcmc._
import io.github.mandar2812.dynaml.probability.GaussianRV
import ammonite.ops._
import ammonite.ops.ImplicitWd._

import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel

import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._


def apply(
  bulk_data_size: Int = 50,
  boundary_data_size: Int = 50,
  basisSize: (Int, Int) = (20, 19),
  reg_data: Double = 0.5,
  reg_galerkin: Double = 1.0,
  burn: Int = 2000,
  num_post_samples: Int = 5000) = {

  measurement_noise = GaussianRV(0.0, 0.5)
  num_bulk_data = bulk_data_size
  num_boundary_data = boundary_data_size

  lambda_params = (-1, 1.5, 0d, -0.4)

  q_params = (-0.5, 1.0d, 0.5, 0.45)

  initialPSD = (l: Double) => {
    val c = utils.chebyshev(3, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
    4000d + 1000*c - 1000*utils.chebyshev(5, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
  }

  nL = 200
  nT = 100

  val rds = RDExperiment.solver(lShellLimits, timeLimits, nL, nT)

  val hybrid_basis = new HybridMQPSDBasis(0.75d)(
    lShellLimits, 14, timeLimits, 19, (false, false)
  )

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_imq_basis(
    0.75, lShellLimits, basisSize._1,
    timeLimits, basisSize._2)


  val seKernel = new GenExpSpaceTimeKernel[Double](
    1d, deltaL, deltaT)(
    sqNormDouble, l1NormDouble)

  val noiseKernel = new DiracTuple2Kernel(1.5)

  noiseKernel.block_all_hyper_parameters

  val (solution, (boundary_data, bulk_data), _) = RDExperiment.generateData(
    rds, dll, lambda, Q, initialPSD)(
    measurement_noise, num_boundary_data,
    num_bulk_data, num_dummy_data)

  RDExperiment.visualisePSD(lShellLimits, timeLimits, nL, nT)(initialPSD, solution, Kp)

  val model = new SGRadialDiffusionModel(
    Kp, dll_params,
    (0d, 0.2, 0d, 0.0),
    (0.01, 0.01d, 0.01, 0.01))(
    seKernel, noiseKernel,
    boundary_data ++ bulk_data,
    chebyshev_hybrid_basis,
    lShellLimits, timeLimits
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(c =>
        c.contains("dll") ||
        c.contains("base::") ||
        c.contains("lambda_gamma")
      )
  }


  model.block(blocked_hyp:_*)

  val hyp = model.effective_hyper_parameters
  val hyper_prior = {
    hyp.filter(_.contains("base::")).map(h => (h, new LogNormal(0d, 2d))).toMap ++
      hyp.filterNot(h => h.contains("base::") || h.contains("tau")).map(h => (h, new Gaussian(0d, 2.5d))).toMap ++
      Map(
        "lambda_alpha" -> new Gaussian(0d, 1d),
        "lambda_beta" -> new LogNormal(0d, 2d),
        "lambda_b" -> new Gaussian(0d, 2.0),
        "Q_alpha" -> new Gaussian(0d, 2d),
        "Q_beta" -> new LogNormal(0d, 2d),
        "Q_gamma" -> new LogNormal(0d, 2d),
        "Q_b" -> new Gaussian(0d, 2d)).filterKeys(
        hyp.contains)
  }

  model.regCol = reg_galerkin
  model.regObs = reg_data

  //Create the MCMC sampler
  val mcmc_sampler = new AdaptiveHyperParameterMCMC[
    model.type, ContinuousDistr[Double]](
    model, hyper_prior, burn)


  //Draw samples from the posterior
  val samples = mcmc_sampler.iid(num_post_samples).draw

  val resPath = RDExperiment.writeResults(
    solution, boundary_data, bulk_data, model.ghost_points,
    hyper_prior, samples, basisSize, "HybridMQ",
    (model.regCol, model.regObs))

  val scriptPath = pwd / "mag-core" / 'scripts / "visualiseCombSamplingResults.R"

  try {
    %%('Rscript, scriptPath.toString, resPath.toString)
  } catch {
    case e: ammonite.ops.ShelloutException => pprint.pprintln(e)
  }

  RDExperiment.visualiseResultsLoss(samples, gt, hyper_prior)
  RDExperiment.visualiseResultsInjection(samples, gt, hyper_prior)

  RDExperiment.samplingReport(
    samples.map(_.filterKeys(quantities_loss.contains)),
    hyp.filter(quantities_loss.contains).map(c => (c, quantities_loss(c))).toMap,
    gt, mcmc_sampler.sampleAcceptenceRate)

  RDExperiment.samplingReport(
    samples.map(_.filterKeys(quantities_injection.contains)),
    hyp.filter(quantities_injection.contains).map(c => (c, quantities_injection(c))).toMap,
    gt, mcmc_sampler.sampleAcceptenceRate, "injection")


  (solution, (boundary_data, bulk_data), model, hyper_prior, mcmc_sampler, samples, resPath)
}

@main
def main(
  bulk_data_size: Int = 50,
  boundary_data_size: Int = 50,
  basisSize: (Int, Int) = (20, 19),
  reg_data: Double = 0.5,
  reg_galerkin: Double = 1.0,
  burn: Int = 2000,
  num_post_samples: Int = 5000) =
  apply(
    bulk_data_size, boundary_data_size,
    basisSize, reg_data, reg_galerkin,
    burn, num_post_samples)