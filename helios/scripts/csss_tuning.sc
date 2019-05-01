import $exec.helios.scripts.csss
import $exec.helios.scripts.csss_pdt_model_tuning
import $exec.helios.scripts.env

val csss_exp = csss_pdt_model_tuning(
  start_year = 2012,
  end_year = 2016,
  test_year = 2015,
  sw_threshold = 600d,
  crop_latitude = 50d,
  fraction_pca = 0.9d,
  fte_step = 24,
  history_fte = 0,
  log_scale_omni = false,
  log_scale_fte = true,
  causal_window = (72, 48),
  network_size = Seq(50, 50, 50),
  activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](1, 2, i, 0f),
  hyper_optimizer = "gs",
  num_samples = 4,
  quantity = OMNIData.Quantities.V_SW,
  reg_type = "L2",
  batch_size = 128,
  max_iterations = 200000,
  max_iterations_tuning = 20000,
  pdt_iterations_tuning = 4,
  pdt_iterations_test = 19,
  optimization_algo = tf.train.Adam(0.001f),
  summary_dir = env.summary_dir,
  get_training_preds = false,
  existing_exp = None //csss.experiments().lastOption
)

try {
  %%(
    'Rscript,
    script,
    csss_exp.results.summary_dir,
    csss.scatter_plots_test(csss_exp.results.summary_dir).last,
    "test_"
  )
} catch {
  case e: Exception => e.printStackTrace()
}

helios.visualise_cdt_results(
  csss.scatter_plots_test(csss_exp.results.summary_dir).last
)
