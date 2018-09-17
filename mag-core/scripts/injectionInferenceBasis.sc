import io.github.mandar2812.dynaml.repl.Router.main
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.probability.mcmc._
import io.github.mandar2812.dynaml.probability.GaussianRV
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.MagParamBasis._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._


def apply(
  bulk_data_size: Int                         = 50,
  boundary_data_size: Int                     = 50,
  basisSize: (Int, Int)                       = (4, 4),
  reg_data: Double                            = 0.5,
  reg_galerkin: Double                        = 1.0,
  burn: Int                                   = 2000,
  num_post_samples: Int                       = 5000,
  lambda_gt: (Double, Double, Double, Double) = lambda_params,
  q_gt: (Double, Double, Double, Double)      = (-0.5, 1.0d, 0.5, 0.45)) = {

  measurement_noise = GaussianRV(0.0, 0.5)
  num_bulk_data = bulk_data_size
  num_boundary_data = boundary_data_size

  lambda_params = lambda_gt

  q_params = q_gt

  initialPSD = (l: Double) => {
    val c = utils.chebyshev(3, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
    4000d + 1000*c - 1000*utils.chebyshev(5, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
  }

  nL = 200
  nT = 100

  val rds = RDExperiment.solver(lShellLimits, timeLimits, nL, nT)

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_laguerre_basis(
    lShellLimits, basisSize._1,
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

  val hyp_basis = Seq("Q_b", "lambda_b").map(
    h => (
      h,
      if(h.contains("_alpha") || h.contains("_b")) hermite_basis(4)
      else if(h.contains("_beta") || h.contains("_gamma")) laguerre_basis(4, 0d)
      else hermite_basis(4)
    )
  ).toMap

  val model = new SGRadialDiffusionModel(
    Kp, dll_params,
    lambda_gt,
    (0.01, 0.01d, 0.01, 0.01))(
    seKernel, noiseKernel,
    boundary_data ++ bulk_data,
    chebyshev_hybrid_basis,
    lShellLimits, timeLimits/*,
    hyper_param_basis = hyp_basis*/
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(c =>
        c.contains("dll") ||
          c.contains("base::") ||
          c.contains("lambda_")
      )
  }


  model.block(blocked_hyp:_*)

  val hyp = model.effective_hyper_parameters

  val h_prior = RDExperiment.hyper_prior(hyp)

  model.regCol = reg_galerkin
  model.regObs = reg_data

  //Create the MCMC sampler
  val mcmc_sampler = new AdaptiveHyperParameterMCMC[
    model.type, ContinuousDistr[Double]](
    model, h_prior, burn)


  //Draw samples from the posterior
  val samples = mcmc_sampler.iid(num_post_samples).draw

  val resPath = RDExperiment.writeResults(
    solution, boundary_data, bulk_data, model.ghost_points,
    h_prior, samples, basisSize, "HybridMQ",
    (model.regCol, model.regObs))

  val scriptPath = pwd / "mag-core" / 'scripts / "visualiseSamplingResults.R"

  try {
    %%('Rscript, scriptPath.toString, resPath.toString, "injection")
  } catch {
    case e: ammonite.ops.ShelloutException => pprint.pprintln(e)
  }

  RDExperiment.visualiseResultsInjection(
    if(num_post_samples > 5000) samples.takeRight(5000) else samples,
    gt, h_prior)

  RDExperiment.samplingReport(
    samples.map(_.filterKeys(quantities_injection.contains)),
    hyp.filter(quantities_injection.contains).map(c => (c, quantities_injection(c))).toMap,
    gt, mcmc_sampler.sampleAcceptenceRate, "injection")


  (solution, (boundary_data, bulk_data), model, h_prior, mcmc_sampler, samples, resPath)
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