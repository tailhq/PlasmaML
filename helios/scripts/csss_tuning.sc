import $exec.helios.scripts.csss_model_tuning
import $exec.helios.scripts.env

import ammonite.ops.ImplicitWd._

val experiments = ls! env.summary_dir |? (_.isDir) |? (_.segments.last.contains("fte"))

val csss_exp = csss_model_tuning(
  start_year = 2008,
  end_year = 2017,
  test_year = 2015,
  sw_threshold = 600d,
  crop_latitude = 30d,
  fte_step = 12,
  history_fte = 6,
  log_scale_omni = false,
  log_scale_fte = false,
  network_size = Seq(40, 50),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  hyper_optimizer = "gs",
  num_samples = 20,
  quantity = OMNIData.Quantities.V_SW,
  reg_type = "L2",
  batch_size = 1000,
  max_iterations = 250000,
  max_iterations_tuning = 20000,
  optimization_algo = tf.train.Adam(0.01f),
  summary_dir = env.summary_dir,
  existing_exp = experiments.lastOption
)

val scatter_plots = ls! csss_exp.results.summary_dir |? (_.segments.last.contains("scatter"))

helios.visualise_cdt_results(scatter_plots.last) 
