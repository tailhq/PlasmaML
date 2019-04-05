import $exec.helios.scripts.csss
import $exec.helios.scripts.csss_model_tuning
import $exec.helios.scripts.env


val csss_exp = csss_model_tuning(
  start_year = 2011,
  end_year = 2016,
  test_year = 2014,
  sw_threshold = 600d,
  crop_latitude = 25d,
  fraction_pca = 0.85,
  fte_step = 24,
  history_fte = 1,
  log_scale_omni = false,
  log_scale_fte = false,
  causal_window = (56, 56),
  network_size = Seq(20, 20),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  hyper_optimizer = "gs",
  num_samples = 16,
  quantity = OMNIData.Quantities.V_SW,
  reg_type = "L2",
  batch_size = 1000,
  max_iterations = 120000,
  max_iterations_tuning = 500,
  optimization_algo = tf.train.Adam(0.01f),
  summary_dir = env.summary_dir,
  get_training_preds = false,
  existing_exp = None//csss.experiments().lastOption
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

helios.visualise_cdt_results(csss.scatter_plots_test(csss_exp.results.summary_dir).last)
