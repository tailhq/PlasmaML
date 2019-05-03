import $exec.helios.scripts.mex_omni
import $exec.helios.scripts.csss

import _root_.io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._

mex_omni.dump_omni_mex_data(
  2010,
  2018,
  (8, 16),
  List(V_SW, /* V_Lat, V_Lon, */ P , B_X, B_Y, B_Z),
  home / 'Downloads / "omni_mex_data.json"
)

val omni_mex_res = mex_omni(
  data_file = home / 'Downloads / "omni_mex_data.json",
  start_year = 2015,
  end_year = 2017,
  test_year = 2016,
  network_size = Seq(20, 20),
  activation_func = (i: Int) => tf.learn.Sigmoid(s"Act_$i"),//timelag.utils.getSinAct[Double](1, i),
  iterations = 100000,
  iterations_tuning = 20000,
  pdt_iterations_tuning = 4,
  pdt_iterations_test = 9,
  batch_size = 32,
  optimizer = tf.train.Adam(0.001f)
)

helios.visualise_cdt_results(
  csss.scatter_plots_test(omni_mex_res.results.summary_dir).last
)
