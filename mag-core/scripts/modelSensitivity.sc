import io.github.mandar2812.dynaml.repl.Router.main
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.probability.mcmc._
import io.github.mandar2812.dynaml.probability.GaussianRV
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.MagRadialDiffusion.{DiffusionField, Injection, LossRate}
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._


def apply(
  lambda_gt: (Double, Double, Double, Double) = (-1, 1.5, 0d, -0.4),
  q_gt: (Double, Double, Double, Double)      = (-0.5, 1.0d, 0.5, 0.45),
  param: String = "dll_beta") = {


  lambda_params = lambda_gt

  q_params = q_gt

  initialPSD = (l: Double) => {
    val c = utils.chebyshev(3, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
    4000d + 1000*c - 1000*utils.chebyshev(5, 2*(l-lShellLimits._1)/(lShellLimits._2 - lShellLimits._1) - 1)
  }

  nL = 200
  nT = 100

  val (diff_process, loss_process, injection_process) = (
    MagTrend(Kp, "dll"),
    MagTrend(Kp, "lambda"),
    MagTrend(Kp, "Q")
  )

  val forward_model = MagRadialDiffusion(
    diff_process,
    loss_process,
    injection_process)(
    lShellLimits, timeLimits,
    nL, nT)

  val solution = forward_model.solve(
    gt.filterKeys(_.contains("dll_")),
    gt.filterKeys(_.contains("lambda_")),
    gt.filterKeys(_.contains("Q_")))(initialPSD)


  RDExperiment.visualisePSD(lShellLimits, timeLimits, nL, nT)(initialPSD, solution, Kp)

  val sensitivity = forward_model.sensitivity(
    DiffusionField(diff_process.transform._keys))(
    gt.filterKeys(_.contains("dll_")),
    gt.filterKeys(_.contains("lambda_")),
    gt.filterKeys(_.contains("Q_")))(_ => 0d)

  RDExperiment.visualisePSD(lShellLimits, timeLimits, nL, nT)(_ => 0d, sensitivity(param), Kp)

  (forward_model, solution, sensitivity)

}

