import breeze.stats.distributions._
import breeze.linalg._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.probability.mcmc._
import io.github.mandar2812.dynaml.probability.GaussianRV
import ammonite.ops._
import ammonite.ops.ImplicitWd._

import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel

import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._

def plot_surrogate(exp_result: RDExperiment.Result[SGRadialDiffusionModel]) = {

  val colocation_points = RDExperiment.readColocation(
    exp_result.results_dir / "colocation_points.csv"
  )

  val (lShellVec, timeVec) =
    RadialDiffusion.buildStencil(lShellLimits, nL, timeLimits, nT)

  val solution_data = timeVec
    .zip(exp_result.solution.map(_.toArray.toSeq).map(lShellVec.zip(_)))
    .flatMap(c => c._2.map(d => ((d._1, c._1), d._2)))
    .toStream

  val solution_data_features = solution_data.map(_._1)

  val solution_targets = solution_data.map(_._2)

  val (params, psi) = exp_result.model.getParams(gt)

  val phi_sol = solution_data_features.map(exp_result.model.basis(_))

  val mean = phi_sol.map(p => {
    val features = DenseVector.vertcat(
      DenseVector(1d),
      exp_result.model.phi * p,
      psi * p
    )
    features.t * params
  })

  val surrogate_preds =
    solution_data
      .zip(
        mean.map(x => x * exp_result.model.psd_std + exp_result.model.psd_mean)
      )
      .map(c => (c._1._1, c._1._2, c._2))

  val metrics = new RegressionMetrics(
    surrogate_preds.map(p => (p._3, p._2)).toList,
    surrogate_preds.length
  )

  (
    surrogate_preds,
    metrics,
    plot3d.draw(surrogate_preds.map(p => (p._1, p._2))),
    plot3d.draw(surrogate_preds.map(p => (p._1, p._3))),
    plot3d.draw(surrogate_preds.map(p => (p._1, p._2 - p._3)))
  )

}
