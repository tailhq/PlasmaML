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
  bulk_data_size: Int = 50,
  boundary_data_size: Int = 50,
  basisSize: (Int, Int) = (4, 4),
  reg: Double = 1d,
  reg_data: Double = 0.5,
  reg_galerkin: Double = 1.0,
  quadrature_l: SGRadialDiffusionModel.QuadratureRule =
    SGRadialDiffusionModel.eightPointGaussLegendre,
  quadrature_t: SGRadialDiffusionModel.QuadratureRule =
    SGRadialDiffusionModel.eightPointGaussLegendre,
  burn: Int = 2000,
  num_post_samples: Int = 5000,
  lambda_gt: (Double, Double, Double, Double) = defaults.lambda_params.values,
  q_gt: (Double, Double, Double, Double) = (-0.5, 1.0d, 0.5, 0.45),
  basisCovFlag: Boolean = true,
  modelType: String = "pure"
): RDExperiment.ResultSynthetic[SGRadialDiffusionModel] = {

  val measurement_noise = GaussianRV(0.0, 1d)
  val num_bulk_data     = bulk_data_size
  val num_boundary_data = boundary_data_size

  val lambda_params = MagParams(lambda_gt)

  val q_params = MagParams(q_gt)

  val RDDomain(lShell, _, time, _) = defaults.rd_domain

  val f0 = (l: Double) => {
    val c = utils.chebyshev(
      3,
      2 * (l - lShell.limits._1) / (lShell.limits._2 - lShell.limits._1) - 1
    )
    2000d + 500 * c - 500 * utils.chebyshev(
      5,
      2 * (l - lShell.limits._1) / (lShell.limits._2 - lShell.limits._1) - 1
    )
  }

  val rds = RDExperiment.solver(
    defaults.rd_domain
  )

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_imq_basis(
    1d,
    lShell.limits,
    basisSize._1,
    time.limits,
    basisSize._2,
    kind = 1
  )

  val seKernel = new GenExpSpaceTimeKernel[Double](
    1d,
    deltaL(defaults.rd_domain),
    deltaT(defaults.rd_domain)
  )(
    sqNormDouble,
    l1NormDouble
  )

  val noiseKernel = new DiracTuple2Kernel(1d)

  noiseKernel.block_all_hyper_parameters

  val (solution, (boundary_data, bulk_data), _) =
    RDExperiment.generateData(
      rds,
      dll(defaults.dll_params, defaults.Kp),
      lambda(lambda_params, defaults.Kp),
      Q(q_params, defaults.Kp, defaults.rd_domain),
      f0
    )(
      measurement_noise,
      num_boundary_data,
      num_bulk_data,
      0
    )

  RDExperiment.visualisePSD(
    defaults.rd_domain,
    f0,
    solution,
    defaults.Kp
  )

  val initial_config = (
    new Uniform(-10d, 10d).draw,
    new Uniform(0d, 10d).draw,
    0d,
    new Uniform(0d, 2d).draw
  )

  val model = if (modelType == "pure") {
    new GalerkinRDModel(
      defaults.Kp,
      defaults.dll_params.values,
      lambda_gt,
      initial_config
    )(
      seKernel,
      noiseKernel,
      boundary_data ++ bulk_data,
      chebyshev_hybrid_basis,
      defaults.rd_domain.l_shell.limits,
      defaults.rd_domain.time.limits,
      quadrature_l,
      quadrature_t,
      basisCovFlag = basisCovFlag
    )
  } else {
    new SGRadialDiffusionModel(
      defaults.Kp,
      defaults.dll_params.values,
      lambda_gt,
      initial_config
    )(
      seKernel,
      noiseKernel,
      boundary_data ++ bulk_data,
      chebyshev_hybrid_basis,
      defaults.rd_domain.l_shell.limits,
      defaults.rd_domain.time.limits,
      quadrature_l,
      quadrature_t,
      basisCovFlag = basisCovFlag
    )
  }

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(
        c =>
          c.contains("dll") ||
            c.contains("base::") ||
            c.contains("lambda_") ||
            c.contains("_gamma")
      )
  }

  model.block(blocked_hyp: _*)

  val hyp = model.effective_hyper_parameters

  val h_prior = RDExperiment.hyper_prior(hyp)

  model.regCol = reg_galerkin
  model.regObs = reg_data
  model.reg = reg

  //Create the MCMC sampler
  val mcmc_sampler =
    new AdaptiveHyperParameterMCMC[SGRadialDiffusionModel, ContinuousDistr[
      Double
    ]](
      model,
      h_prior,
      burn
    )

  //Draw samples from the posterior
  val samples = mcmc_sampler.iid(num_post_samples).draw

  val resPath = RDExperiment.writeResults(
    DataConfig(
      defaults.rd_domain,
      defaults.dll_params,
      lambda_params,
      q_params,
      f0,
      defaults.Kp,
      measurement_noise
    ),
    solution,
    boundary_data,
    bulk_data,
    model.ghost_points,
    h_prior,
    samples,
    basisSize,
    s"ChebyshevIMQ[beta=1]",
    (model.regCol, model.regObs)
  )

  val scriptPath = pwd / "mag-core" / 'scripts / "visualiseSamplingResults.R"

  try {
    %%('Rscript, scriptPath.toString, resPath.toString, "injection")
  } catch {
    case e: ammonite.ops.ShelloutException => pprint.pprintln(e)
  }

  val ground_truth_map = gt(defaults.dll_params, lambda_params, q_params)

  RDExperiment.visualiseResultsInjection(
    if (num_post_samples > 5000) samples.takeRight(5000) else samples,
    ground_truth_map,
    h_prior
  )

  RDExperiment.samplingReport(
    samples.map(_.filterKeys(quantities_injection.contains)),
    hyp
      .filter(quantities_injection.contains)
      .map(c => (c, quantities_injection(c)))
      .toMap,
    ground_truth_map,
    mcmc_sampler.sampleAcceptenceRate,
    "injection"
  )

  RDExperiment.ResultSynthetic(
    DataConfig(
      defaults.rd_domain,
      defaults.dll_params,
      lambda_params,
      q_params,
      f0,
      defaults.Kp,
      measurement_noise
    ),
    solution,
    (boundary_data, bulk_data),
    model,
    h_prior,
    mcmc_sampler,
    samples,
    resPath
  )
}

@main
def main(
  bulk_data_size: Int = 50,
  boundary_data_size: Int = 50,
  basisSize: (Int, Int) = (20, 19),
  reg_data: Double = 0.5,
  reg_galerkin: Double = 1.0,
  burn: Int = 2000,
  num_post_samples: Int = 5000
) =
  apply(
    bulk_data_size,
    boundary_data_size,
    basisSize,
    reg_data,
    reg_galerkin,
    burn = burn,
    num_post_samples = num_post_samples
  )
