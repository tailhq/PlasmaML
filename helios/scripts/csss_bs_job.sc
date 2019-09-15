import _root_.io.github.mandar2812.dynaml.{utils => dutils}
import $file.csss
import $file.csss_pdt_model_tuning
import $file.csss_so_tuning
import $file.env
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time._
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.io.github.mandar2812.PlasmaML.omni.OMNIData

@main
def main(
  csss_job_id: String,
  exp_dir: String,
  network_size: Seq[Int] = Seq(50, 50),
  root_dir: String = env.summary_dir.toString
) = {

  val csss_fixed = csss_so_tuning.baseline(
    Path(s"${root_dir}/${csss_job_id}/${exp_dir}"),
    network_size = network_size,
    activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
    optimization_algo = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    max_iterations = csss.ext_iterations,
    max_iterations_tuning = csss.base_iterations,
    batch_size = 128,
    num_samples = 4,
    data_scaling = "hybrid",
    use_copula = true
  )

  println("Base Line Model Performance:")
  csss_fixed.results.metrics_test.get.print()

}
