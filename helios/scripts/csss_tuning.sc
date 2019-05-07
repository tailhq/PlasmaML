import $exec.helios.scripts.csss
import $exec.helios.scripts.csss_pdt_model_tuning
import $exec.helios.scripts.env


val time_window = (72, 48)
val avg_sw_6h = DataPipe((xs: Seq[Double]) => xs.grouped(6).map(g => g.sum/g.length).toSeq)
val max_sw_6h = DataPipe((xs: Seq[Double]) => xs.grouped(6).map(g => g.max).toSeq)

val fact = 20
val base_iterations = 40000
def ext_iterations  = base_iterations * fact

val base_it_pdt     = 4
def ext_it_pdt      = 2*base_it_pdt + 1

val csss_exp = csss_pdt_model_tuning(
  start_year = 2010,
  end_year = 2016,
  test_year = 2015,
  crop_latitude = 25d,
  fraction_pca = 1d,
  fte_step = 0,
  history_fte = 0,
  log_scale_omni = false,
  log_scale_fte = true,
  time_window = time_window,
  ts_transform_output = avg_sw_6h,
  network_size = Seq(50, 50),
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

helios.visualise_cdt_results(
  csss.scatter_plots_test(csss_exp.results.summary_dir).last
)
