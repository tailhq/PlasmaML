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
  lambda_gt: (Double, Double, Double, Double) = (math.log(math.pow(10d,
        -4) * math.pow(10d, 2.5d) / 2.4), 1d, 0d, 0.18),
  q_gt: (Double, Double, Double, Double) = (-0.5, 1.0d, 0.5, 0.45),
  basisCovFlag: Boolean = true,
  modelType: String = "pure"
): RDExperiment.ResultSynthetic[SGRadialDiffusionModel] = {

  measurement_noise = GaussianRV(0.0, 1d)
  num_bulk_data = bulk_data_size
  num_boundary_data = boundary_data_size

  lambda_params = lambda_gt

  q_params = q_gt

  initialPSD = (l: Double) => {
    val c = utils.chebyshev(
      3,
      2 * (l - lShellLimits._1) / (lShellLimits._2 - lShellLimits._1) - 1
    )
    4000d + 1000 * c - 1000 * utils.chebyshev(
      5,
      2 * (l - lShellLimits._1) / (lShellLimits._2 - lShellLimits._1) - 1
    )
  }

  nL = 200
  nT = 100

  val rds = RDExperiment.solver(lShellLimits, timeLimits, nL, nT)

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_imq_basis(
    1d,
    lShellLimits,
    basisSize._1,
    timeLimits,
    basisSize._2,
    kind = 1
  )

  val seKernel = new GenExpSpaceTimeKernel[Double](1d, deltaL, deltaT)(
    sqNormDouble,
    l1NormDouble
  )

  val noiseKernel = new DiracTuple2Kernel(1d)

  noiseKernel.block_all_hyper_parameters

  val (solution, (boundary_data, bulk_data), _) =
    RDExperiment.generateData(rds, dll, lambda, Q, initialPSD)(
      measurement_noise,
      num_boundary_data,
      num_bulk_data,
      num_dummy_data
    )

  RDExperiment.visualisePSD(lShellLimits, timeLimits, nL, nT)(
    initialPSD,
    solution,
    Kp
  )

  
  val initial_config = (
    new Uniform(-10d, 10d).draw,
    new Uniform(0d, 10d).draw,
    0d,
    new Uniform(0d, 2d).draw
  )

  val model = if (modelType == "pure") {
    new GalerkinRDModel(
      Kp,
      dll_params,
      lambda_gt,
      initial_config
    )(
      seKernel,
      noiseKernel,
      boundary_data ++ bulk_data,
      chebyshev_hybrid_basis,
      lShellLimits,
      timeLimits,
      quadrature_l,
      quadrature_t,
      basisCovFlag = basisCovFlag
    )
  } else {
    new SGRadialDiffusionModel(
      Kp,
      dll_params,
      lambda_gt,
      initial_config
    )(
      seKernel,
      noiseKernel,
      boundary_data ++ bulk_data,
      chebyshev_hybrid_basis,
      lShellLimits,
      timeLimits,
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
  model.reg    = reg

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

  RDExperiment.visualiseResultsInjection(
    if (num_post_samples > 5000) samples.takeRight(5000) else samples,
    gt,
    h_prior
  )

  RDExperiment.samplingReport(
    samples.map(_.filterKeys(quantities_injection.contains)),
    hyp
      .filter(quantities_injection.contains)
      .map(c => (c, quantities_injection(c)))
      .toMap,
    gt,
    mcmc_sampler.sampleAcceptenceRate,
    "injection"
  )

  RDExperiment.ResultSynthetic(
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
