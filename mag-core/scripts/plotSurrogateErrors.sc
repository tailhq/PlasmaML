{
  import breeze.stats.distributions._
  import breeze.linalg._
  import io.github.mandar2812.dynaml.kernels._
  import io.github.mandar2812.dynaml.probability.mcmc._
  import io.github.mandar2812.dynaml.probability.GaussianRV
  import ammonite.ops._
  import ammonite.ops.ImplicitWd._

  import io.github.mandar2812.PlasmaML.dynamics.diffusion._
  import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel

  import io.github.mandar2812.PlasmaML.dynamics.diffusion._
  import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._

  val resPath = pwd/".cache"/"radial-diffusion-exp_2017_10_31_12_45"

  val (solution, (boundary_data, bulk_data), colocation_points, samples, basisInfo) =
    RDExperiment.loadCachedResults(resPath)

  val (lShellVec, timeVec) = RadialDiffusion.buildStencil(lShellLimits, nL, timeLimits, nT)

  val solution_data = timeVec.zip(
    solution.map(_.toArray.toSeq).map(
    lShellVec.zip(_))).flatMap(
    c => c._2.map(d => ((d._1, c._1), d._2))).toStream

  val solution_data_features = solution_data.map(_._1)

  val solution_targets = solution_data.map(_._2)

  val basisSize = basisInfo._2
  val hybrid_basis = new HybridMQPSDBasis(0.75d)(
    lShellLimits, basisSize._1, timeLimits, basisSize._2, (false, false)
  )

  val seKernel = new GenExpSpaceTimeKernel[Double](
    10d, deltaL, deltaT)(
    sqNormDouble, l1NormDouble)

  val noiseKernel = new DiracTuple2Kernel(1.5)

  noiseKernel.block_all_hyper_parameters

  val model = new BasisFuncRadialDiffusionModel(
    Kp, dll_params,
    (0d, 0.2, 0d, 0.0),
    (0.01, 0.01d, 0.01, 0.01))(
    seKernel, noiseKernel,
    boundary_data ++ bulk_data, colocation_points,
    hybrid_basis
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(c =>
        c.contains("dll") ||
          c.contains("base::") ||
          c.contains("tau_gamma")
      )
  }


  model.block(blocked_hyp:_*)

  val (params, psi) = model.getGalerkinParams(gt)



  val phi_sol = solution_data_features.map(hybrid_basis(_))

  val mean = phi_sol.map(p => {
    val features = DenseVector.vertcat(
      DenseVector(1d),
      model.phi*p,
      psi*p
    )
    features.t*params
  })


  val surrogate_preds = solution_data.zip(mean).map(c => (c._1._1, c._1._2, c._2))
}
