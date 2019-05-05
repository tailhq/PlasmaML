import $exec.helios.scripts.omni_pdt
import $exec.helios.scripts.csss

import _root_.io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._

val omni_res = omni_pdt(
  List(V_SW, B_Z),
  Dst,
  causal_window = (0, 12),
  start_year = 2003,
  end_year = 2010,
  test_year = 2003,
  network_size = Seq(10, 10),
  activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  iterations = 100000,
  iterations_tuning = 20000,
  pdt_iterations_tuning = 4,
  pdt_iterations_test = 4,
  batch_size = 128,
  optimizer = tf.train.Adam(0.001f)
)

helios.visualise_cdt_results(
  csss.scatter_plots_test(omni_res.results.summary_dir).last
)
