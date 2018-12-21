import _root_.ammonite.ops._
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelagutils
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.dynaml.tensorflow.data.DataSet
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.PlasmaML.helios
import org.joda.time.DateTime
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer}

type CV_RESULT = Map[
  Int,
  helios.ExperimentResult[
    DataSet[(DateTime, (Tensor, Tensor))],
    Tensor, Tensor,
    (Tensor, Tensor),
    (Output, Output)]
  ]

type CV_RESULT_SO = Map[
  Int,
  helios.ExperimentResult[
    DataSet[(DateTime, (Tensor, Tensor))],
    Tensor, Tensor,
    Tensor, Output]
  ]


@main
def apply(
    start_year: Int = 2011,
    end_year: Int = 2017, 
    divergence_term: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.JensenShannon,
    temperature: Double = 1.5,
    network_size: Seq[Int] = Seq(100, 60),
    activation_func: Int => Activation = timelagutils.getReLUAct(1),
    history_fte: Int = 10,
    fte_step: Int = 2,
    crop_latitude: Double = 40d, 
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    causal_window: (Int, Int) = (48, 72),
    max_iterations: Int = 100000, 
    batch_size: Int = 32, 
    regularization_const: Double = 0.001, 
    optimization_algo: tf.train.Optimizer = tf.train.Adam(0.01),
    summary_dir: Path = home/'tmp): CV_RESULT = {
    
    val cv_results = (start_year to end_year).map(ty => {

        val result = fte.exp_cdt(
          year_range = start_year to end_year,
          test_year = ty,
          optimizer = optimization_algo,
          num_neurons = network_size,
          activation_func = activation_func,
          miniBatch = batch_size,
          iterations = max_iterations,
          latitude_limit = crop_latitude,
          deltaTFTE = history_fte,
          fteStep = fte_step,
          log_scale_fte = log_scale_fte,
          log_scale_omni = log_scale_omni,
          deltaT = causal_window,
          reg = regularization_const,
          divergence = divergence_term,
          summary_top_dir = summary_dir,
          temperature = temperature
        )
        
        fte.FTExperiment.clear_cache()
        (ty, result)
    }).toMap

  cv_results

}

def single_output(
  start_year: Int = 2011,
  end_year: Int = 2017,
  network_size: Seq[Int] = Seq(100, 80),
  activation_func: Int => Activation = timelagutils.getReLUAct(1),
  history_fte: Int = 10,
  fte_step: Int = 2,
  crop_latitude: Double = 40d,
  log_scale_fte: Boolean = false,
  log_scale_omni: Boolean = false,
  causal_lag: Int = 96,
  max_iterations: Int = 100000,
  batch_size: Int = 32,
  regularization_const: Double = 0.001,
  optimization_algo: tf.train.Optimizer = tf.train.Adam(0.01),
  summary_dir: Path = home/'tmp): CV_RESULT_SO = {

    val cv_results = (start_year to end_year).map(ty => {

        val result = fte.exp_single_output(
          year_range = start_year to end_year,
          test_year = ty,
          optimizer = optimization_algo,
          num_neurons = network_size,
          activation_func = activation_func,
          miniBatch = batch_size,
          iterations = max_iterations,
          latitude_limit = crop_latitude,
          deltaTFTE = history_fte,
          fteStep = fte_step,
          log_scale_fte = log_scale_fte,
          log_scale_omni = log_scale_omni,
          deltaT = causal_lag,
          reg = regularization_const,
          summary_top_dir = summary_dir
        )

        (ty, result)
    }).toMap

    fte.FTExperiment.clear_cache()

    cv_results

}



//cv_results.toSeq.sortBy(_._1).foreach(kv => {
//    kv._2.results.metrics_test.get.generatePlots()
//})
