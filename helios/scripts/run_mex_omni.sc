import $exec.helios.scripts.mex_omni
import $exec.helios.scripts.csss

mex_omni.dump_omni_mex_data(
  2010,
  2018,
  (8, 16),
  home / 'Downloads / "omni_mex_data.json"
)

val omni_mex_res = mex_omni(
  data_file = home / 'Downloads / "omni_mex_data.json",
  start_year = 2014,
  end_year = 2017,
  test_year = 2016,
  network_size = Seq(20, 20),
  activation_func = (i: Int) => tf.learn.Sigmoid(s"Act_$i"),//timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  iterations = 20000,
  iterations_tuning = 8000,
  pdt_iterations_tuning = 2,
  pdt_iterations_test = 4,
  batch_size = 128,
  optimizer = tf.train.Adam(0.001f)
)

helios.visualise_cdt_results(
  csss.scatter_plots_test(omni_mex_res.results.summary_dir).last
)
