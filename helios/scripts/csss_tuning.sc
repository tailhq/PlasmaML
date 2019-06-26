import _root_.io.github.mandar2812.dynaml.{utils => dutils}
import $exec.helios.scripts.csss
import $exec.helios.scripts.csss_pdt_model_tuning
import $exec.helios.scripts.csss_so_tuning
import $exec.helios.scripts.env

val time_window = (48, 72)

val avg_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => g.sum / g.length).toSeq
)

val max_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => g.max).toSeq
)

val median_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => dutils.median(g.toStream)).toSeq
)


val avg_sw_24h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(24).map(g => g.sum / g.length).toSeq
)

val max_sw_24h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(24).map(g => g.max).toSeq
)

val median_sw_24h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(24).map(g => dutils.median(g.toStream)).toSeq
)

val fact            = 10
val base_iterations = 80000
def ext_iterations  = base_iterations * fact

val base_it_pdt = 3
def ext_it_pdt  = base_it_pdt + 2

val csss_exp = csss_pdt_model_tuning(
  start_year = 2010,
  end_year = 2018,
  test_year = 2018,
  test_month = 10,
  crop_latitude = 90d,
  sw_threshold = 600d,
  fraction_pca = 1d,
  fte_step = 0,
  history_fte = 0,
  log_scale_omni = false,
  log_scale_fte = true,
  time_window = time_window,
  ts_transform_output = median_sw_6h,
  network_size = Seq(60, 40),
  use_persistence = true,
  activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  hyper_optimizer = "gs",
  num_samples = 4,
  quantity = OMNIData.Quantities.V_SW,
  reg_type = "L2",
  batch_size = 128,
  max_iterations = ext_iterations,
  max_iterations_tuning = base_iterations,
  pdt_iterations_tuning = base_it_pdt,
  pdt_iterations_test = ext_it_pdt,
  optimization_algo = tf.train.Adam(0.001f),
  summary_dir = env.summary_dir,
  get_training_preds = true,
  data_scaling = "gauss",
  use_copula = true,
  existing_exp = None
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

val csss_fixed = csss_so_tuning.baseline(
  csss_exp.results.summary_dir,
  network_size = Seq(60, 40),
  activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  optimization_algo = tf.train.Adam(0.001f),
  max_iterations = ext_iterations,
  max_iterations_tuning = base_iterations,
  batch_size = 512,
  num_samples = 4
)
