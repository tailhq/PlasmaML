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
import org.joda.time._
import org.joda.time.format.DateTimeFormat
import ammonite.ops.ImplicitWd._

case class RDSensitivity(
  config: DataConfig,
  forward_model: MagRadialDiffusion[Map[String, Double]],
  psd: Seq[((Double, Double), Double)],
  sensitivity: Map[String, Seq[((Double, Double), Double)]],
  plots: Map[String, org.jzy3d.analysis.AbstractAnalysis])

def write_results(results: RDSensitivity): Unit = {

  val DataConfig(
    rd_domain,
    dll_params,
    lambad_params,
    q_params,
    initial_psd,
    k_p,
    measurement_noise
  ) = results.config

  //Create a dump folder
  val dateTime = new DateTime()

  val dtString =
    dateTime.toString(DateTimeFormat.forPattern("yyyy_MM_dd_H_mm"))

  val host: Option[String] = try {
    Some(
      java.net.InetAddress
        .getLocalHost()
        .toString
        .split('/')
        .head
        .split('.')
        .head
    )
  } catch {
    case _: java.net.UnknownHostException => None
    case _: Exception                     => None
  }

  val hostStr: String = host match {
    case None    => ""
    case Some(h) => h + "_"
  }

  val resultsPath = pwd / ".cache" / (s"${hostStr}rd_sensitivity_exp_${dtString}")

  if (!exists(resultsPath)) mkdir ! resultsPath
  //write the psd solution
  write(
    resultsPath / "solution.csv",
    "l,t,psd\n" ++
      results.psd.map(p => s"${p._1._1},${p._1._2},${p._2}").mkString("\n")
  )

  //write domain info
  write(
    resultsPath / "diffusion_domain.csv",
    domainSpec(rd_domain).keys
      .mkString(",") + "\n" + domainSpec(rd_domain).values
      .mkString(",")
  )

  val (lShellVec, timeVec) =
    RadialDiffusion.buildStencil(
      rd_domain.l_shell.limits,
      rd_domain.nL,
      rd_domain.time.limits,
      rd_domain.nT
    )

  //val initial_condition = lShellVec.map(l => Seq(l, initial_psd(l)))
  //write Kp profile
  write(
    resultsPath / "kp_profile.csv",
    timeVec.map(t => Seq(t, k_p(t))).map(_.mkString(",")).mkString("\n")
  )

  //write hyper-parameters.
  val ground_truth = gt(dll_params, lambad_params, q_params)

  write(
    resultsPath / "diffusion_params.csv",
    ground_truth.keys.mkString(",") + "\n" + ground_truth.values
      .mkString(",") + "\n"
  )

  //write sensitivity data.
  write(
    resultsPath / "sensitivity.csv",
    "l,t,value,quantity,parameter\n" ++
      results.sensitivity
        .flatMap(kv => {
          val (quantity, param) = (kv._1.split("_").head, kv._1.split("_").last)
          kv._2.map(p => s"${p._1._1},${p._1._2},${p._2},${quantity},${param}")
        })
        .mkString("\n")
  )

  //run plotting script
  val scriptPath = pwd / "mag-core" / 'scripts / "visualiseSensitivity.R"

  try {
    %%('Rscript, scriptPath.toString, resultsPath.toString)
  } catch {
    case e: ammonite.ops.ShelloutException => pprint.pprintln(e)
  }
}

def apply(
  lambda_p: (Double, Double, Double, Double) = defaults.lambda_params.values,
  q_p: (Double, Double, Double, Double) = defaults.q_params.values,
  num_bins_l: Int = 50,
  num_bins_t: Int = 100,
  param: String = "dll_beta"
) = {

  val lambda_params = MagParams(lambda_p)

  val q_params = MagParams(q_p)

  val rd_domain = defaults.rd_domain.copy(nL = num_bins_l, nT = num_bins_t)

  val f0 = (l: Double) => {
    val c = utils.chebyshev(
      3,
      2 * (l - rd_domain.l_shell.limits._1) / (rd_domain.l_shell.limits._2 - rd_domain.l_shell.limits._1) - 1
    )
    2000d + 500 * c - 500 * utils.chebyshev(
      5,
      2 * (l - rd_domain.l_shell.limits._1) / (rd_domain.l_shell.limits._2 - rd_domain.l_shell.limits._1) - 1
    )
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
      "psd" -> plot3d.draw(
        to_pairs(solution).map(c => (c._1, math.log10(c._2)))
      ),
      "dll" -> plot3d.draw(
        dll(defaults.dll_params, defaults.Kp) _,
        (
          rd_domain.l_shell.limits._1.toFloat,
          rd_domain.l_shell.limits._2.toFloat
        ),
        (rd_domain.time.limits._1.toFloat, rd_domain.time.limits._2.toFloat)
      ),
      "lambda" -> plot3d.draw(
        lambda(lambda_params, defaults.Kp) _,
        (
          rd_domain.l_shell.limits._1.toFloat,
          rd_domain.l_shell.limits._2.toFloat
        ),
        (rd_domain.time.limits._1.toFloat, rd_domain.time.limits._2.toFloat)
      ),
      "Q" -> plot3d.draw(
        Q(q_params, defaults.Kp, rd_domain) _,
        (
          rd_domain.l_shell.limits._1.toFloat,
          rd_domain.l_shell.limits._2.toFloat
        ),
        (rd_domain.time.limits._1.toFloat, rd_domain.time.limits._2.toFloat)
      )
    ) ++
      sensitivity.map(
        kv => (kv._1, plot3d.draw(to_pairs(kv._2)))
      )

  RDSensitivity(
    DataConfig(
      rd_domain,
      defaults.dll_params,
      lambda_params,
      q_params,
      f0,
      defaults.Kp,
      GaussianRV(0d, 0d)
    ),
    forward_model,
    to_pairs(solution),
    sensitivity.mapValues(to_pairs),
    plots
  )

}
