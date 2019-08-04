import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.graphics.plot3d
import io.github.mandar2812.PlasmaML.dynamics.diffusion.MagRadialDiffusion.{
  DiffusionField,
  Injection,
  LossRate
}
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._

def apply(
  lambda_gt: (Double, Double, Double, Double) = (-1, 1.5, 0d, -0.4),
  q_gt: (Double, Double, Double, Double) = (-0.5, 1.0d, 0.5, 0.45),
  param: String = "dll_beta"
) = {

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

  val (diff_process, loss_process, injection_process) = (
    MagTrend(Kp, "dll"),
    MagTrend(Kp, "lambda"),
    MagTrend(Kp, "Q")
  )

  val forward_model = MagRadialDiffusion(
    diff_process,
    loss_process,
    injection_process
  )(lShellLimits, timeLimits, nL, nT)

  val (diff_params_map, loss_params_map, inj_params_map) = (
    gt.filterKeys(_.contains("dll_")),
    gt.filterKeys(_.contains("lambda_")),
    gt.filterKeys(_.contains("Q_"))
  )

  val solution =
    forward_model.solve(diff_params_map, loss_params_map, inj_params_map)(
      initialPSD
    )

  val sensitivity_diff = forward_model.sensitivity(
    DiffusionField(diff_process.transform._keys)
  )(diff_params_map, loss_params_map, inj_params_map)(_ => 0d)

  val sensitivity_loss = forward_model.sensitivity(
    LossRate(loss_process.transform._keys)
  )(diff_params_map, loss_params_map, inj_params_map)(_ => 0d)

  val sensitivity_inj = forward_model.sensitivity(
    Injection(injection_process.transform._keys)
  )(diff_params_map, loss_params_map, inj_params_map)(_ => 0d)

  val sensitivity = sensitivity_diff ++ sensitivity_loss ++ sensitivity_inj

  val to_pairs =
    RadialDiffusion.to_input_output_pairs(lShellLimits, timeLimits, nL, nT) _

  val plots =
    Map("psd" -> plot3d.draw(to_pairs(solution))) ++
      sensitivity.map(
        kv => ("sensitivity_" + kv._1, plot3d.draw(to_pairs(kv._2)))
      )

  (forward_model, solution, sensitivity, plots)

}
