import $exec.helios.scripts.mex_omni
import $exec.helios.scripts.csss

import _root_.io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._

mex_omni.dump_omni_mex_data(
  2010,
  2018,
  (8, 16),
  home / 'Downloads / "omni_mex_data.json",
  List(V_SW, V_Lat, V_Lon/* , B_X, B_Y, B_Z */)
)

val omni_mex_res = mex_omni(
  data_file = home / 'Downloads / "omni_mex_data.json",
  start_year = 2014,
  end_year = 2015,
  test_year = 2015,
  network_size = Seq(30, 10),
  activation_func = (i: Int) => dtflearn.Tanh(s"Act_$i"),//timelag.utils.getReLUAct3[Double](1, 2, i, 0f),
  iterations = 100000,
  iterations_tuning = 20000,
  pdt_iterations_tuning = 4,
  pdt_iterations_test = 9,
  batch_size = 128,
  optimizer = tf.train.Adam(0.01f)
)

helios.visualise_cdt_results(
  csss.scatter_plots_test(omni_mex_res.results.summary_dir).last
)
