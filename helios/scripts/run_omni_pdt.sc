import $exec.helios.scripts.omni_pdt
import $exec.helios.scripts.csss

import _root_.io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._

val omni_res = omni_pdt(
  start_year = 2015,
  end_year = 2017,
  test_year = 2016,
  network_size = Seq(20, 20),
  activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  iterations = 100000,
  iterations_tuning = 20000,
  pdt_iterations_tuning = 4,
  pdt_iterations_test = 9,
  batch_size = 128,
  optimizer = tf.train.Adam(0.01f)
)

helios.visualise_cdt_results(
  csss.scatter_plots_test(omni_res.results.summary_dir).last
)
