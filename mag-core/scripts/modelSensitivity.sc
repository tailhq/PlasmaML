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
  lambda: (Double, Double, Double, Double) = defaults.lambda_params.values,
  q: (Double, Double, Double, Double) = defaults.q_params.values,
  num_bins_l: Int = 50,
  num_bins_t: Int = 100,
  param: String = "dll_beta"
) = {

  val lambda_params = MagParams(lambda)

  val q_params = MagParams(q)

  val rd_domain = defaults.rd_domain.copy(nL = num_bins_l, nT = num_bins_t)

  val f0 = (l: Double) => {
    val c = utils.chebyshev(
      2,
      2 * (l - rd_domain.l_shell.limits._1) / (rd_domain.l_shell.limits._2 - rd_domain.l_shell.limits._1) - 1,
      kind = 1
    )
    101d - 100 * c
  }

  val (diff_process, loss_process, injection_process) = (
    MagTrend(defaults.Kp, "dll"),
    MagTrend(defaults.Kp, "lambda"),
    BoundaryInjection(defaults.Kp, rd_domain.l_shell.limits._2, "Q")
  )

  val forward_model = MagRadialDiffusion(
    diff_process,
    loss_process,
    injection_process
  )(rd_domain.l_shell.limits, rd_domain.time.limits, rd_domain.nL, rd_domain.nT)

  val ground_truth_map = gt(defaults.dll_params, lambda_params, q_params)

  val (diff_params_map, loss_params_map, inj_params_map) = (
    ground_truth_map.filterKeys(_.contains("dll_")),
    ground_truth_map.filterKeys(_.contains("lambda_")),
    ground_truth_map.filterKeys(_.contains("Q_"))
  )

  val solution =
    forward_model.solve(diff_params_map, loss_params_map, inj_params_map)(
      f0
    )

  val sensitivity_diff = forward_model.sensitivity(
    DiffusionField(diff_process.transform._keys)
  )(diff_params_map, loss_params_map, inj_params_map)(f0)

  val sensitivity_loss = forward_model.sensitivity(
    LossRate(loss_process.transform._keys)
  )(diff_params_map, loss_params_map, inj_params_map)(f0)

  val sensitivity_inj = forward_model.sensitivity(
    Injection(injection_process.transform._keys)
  )(diff_params_map, loss_params_map, inj_params_map)(f0)

  val sensitivity = sensitivity_diff ++ sensitivity_loss ++ sensitivity_inj

  val to_pairs =
    RadialDiffusion.to_input_output_pairs(
      rd_domain.l_shell.limits,
      rd_domain.time.limits,
      rd_domain.nL,
      rd_domain.nT
    ) _

  val plots =
    Map(
      "psd" -> plot3d
        .draw(to_pairs(solution).map(c => (c._1, math.log10(c._2))))
    ) ++
      sensitivity.map(
        kv => ("sensitivity_" + kv._1, plot3d.draw(to_pairs(kv._2)))
      )

  (
    forward_model,
    to_pairs(solution),
    sensitivity.mapValues(to_pairs),
    plots
  )

}
