{

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

  num_bulk_data = 160
  num_boundary_data = 40

  num_dummy_data = 50

  lambda_params = (
    -1, 2d, 0d, 0.25)


  nL = 300
  nT = 200

  q_params = (2d, 0.5d, 0.05, 0.45)

  initialPSD = (l: Double) => {
    val c = utils.chebyshev(3, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
    4000d + 1000*c - 1000*utils.chebyshev(5, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
  }

  val rds = RDExperiment.solver(lShellLimits, timeLimits, nL, nT)


  val basisSize = (4, 19)
  val hybrid_basis = new HybridMQPSDBasis(0.75d)(
    lShellLimits, basisSize._1, timeLimits, basisSize._2, (false, false)
  )

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_imq_basis(
    0.75, lShellLimits, basisSize._1,
    timeLimits, basisSize._2)

  val burn = 1500

  val seKernel = new GenExpSpaceTimeKernel[Double](
    10d, deltaL, deltaT)(
    sqNormDouble, l1NormDouble)



  val scaledSEKernel = ScaledKernel(seKernel, DataPipe((x: (Double, Double)) => math.abs(x._1)))

  val noiseKernel = new DiracTuple2Kernel(1.5)

  noiseKernel.block_all_hyper_parameters

  val (solution, (boundary_data, bulk_data), colocation_points) = RDExperiment.generateData(
    rds, dll, lambda, Q, initialPSD)(
    measurement_noise, num_boundary_data,
    num_bulk_data, num_dummy_data)

  val model = new BasisFuncRadialDiffusionModel(
    Kp, dll_params, (0d, 0.2, 0d, 0.0), q_params)(
    scaledSEKernel, noiseKernel,
    boundary_data ++ bulk_data, colocation_points,
    chebyshev_hybrid_basis
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(c =>
        c.contains("dll") ||
        c.contains("base::") ||
        c.contains("tau_gamma") ||
        c.contains("Q_")
      )
  }


  model.block(blocked_hyp:_*)

  val eff_hyp = model.effective_hyper_parameters
  val hyper_prior = {
    eff_hyp.filter(_.contains("base::")).map(h => (h, new LogNormal(0d, 2d))).toMap ++
      eff_hyp.filterNot(h => h.contains("base::") || h.contains("tau")).map(h => (h, new Gaussian(0d, 2.5d))).toMap ++
      Map(
        "tau_alpha" -> new Gaussian(0d, 2d),
        "tau_beta" -> new LogNormal(0d, 2d),
        "tau_b" -> new Gaussian(0d, 2.0)).filterKeys(eff_hyp.contains)
  }

  model.regCol = regColocation
  model.regObs = 1E-4

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
}
