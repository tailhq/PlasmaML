import $exec.helios.scripts.csss_model_tuning
import $exec.helios.scripts.env

import ammonite.ops.ImplicitWd._

def experiments() =
  ls ! env.summary_dir |? (_.isDir) |? (_.segments.last.contains("fte"))

val csss_exp = csss_model_tuning(
  start_year = 2008,
  end_year = 2017,
  test_year = 2010,
  sw_threshold = 600d,
  crop_latitude = 25d,
  fraction_pca = 0.8,
  fte_step = 2,
  history_fte = 12,
  log_scale_omni = false,
  log_scale_fte = false,
  causal_window = (56, 56),
  network_size = Seq(30, 30),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  hyper_optimizer = "gs",
  num_samples = 16,
  quantity = OMNIData.Quantities.V_SW,
  reg_type = "L2",
  batch_size = 1000,
  max_iterations = 50000,
  max_iterations_tuning = 500,
  optimization_algo = tf.train.Adam(0.001f),
  summary_dir = env.summary_dir,
  get_training_preds = false,
  existing_exp = experiments().lastOption
)

def scatter_plots_test() =
  ls ! csss_exp.results.summary_dir |? (_.segments.last.contains("scatter_test"))

def scatter_plots_train() =
  ls ! csss_exp.results.summary_dir |? (_.segments.last.contains("scatter_train"))

val script = pwd / 'helios / 'scripts / "visualise_tl.R"

try {
  %%(
    'Rscript,
    script,
    csss_exp.results.summary_dir,
    scatter_plots_test().last,
    "test_"
  )
} catch {
  case e: Exception => e.printStackTrace()
}

helios.visualise_cdt_results(scatter_plots_test().last)
