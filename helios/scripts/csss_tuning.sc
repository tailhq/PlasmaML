import $exec.helios.scripts.csss_model_tuning
import $exec.helios.scripts.env

import ammonite.ops.ImplicitWd._

def experiments() = ls! env.summary_dir |? (_.isDir) |? (_.segments.last.contains("fte"))

val csss_exp = csss_model_tuning(
  start_year = 2008,
  end_year = 2014,
  test_year = 2009,
  sw_threshold = 600d,
  crop_latitude = 25d,
  fraction_pca = 0.8,
  fte_step = 24,
  history_fte = 5,
  log_scale_omni = false,
  log_scale_fte = false,
  causal_window = (56, 56),
  network_size = Seq(50, 50),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  hyper_optimizer = "gs",
  num_samples = 20,
  quantity = OMNIData.Quantities.V_SW,
  reg_type = "L2",
  batch_size = 500,
  max_iterations = 10000,
  max_iterations_tuning = 500,
  optimization_algo = tf.train.Adam(0.001f),
  summary_dir = env.summary_dir,
  get_training_preds = false,
  existing_exp = experiments().headOption
)

def scatter_plots_test() = ls! csss_exp.results.summary_dir |? (_.segments.last.contains("scatter_test"))

helios.visualise_cdt_results(scatter_plots_test().last)
