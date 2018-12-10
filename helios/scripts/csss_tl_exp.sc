import $file.timelagutils
import $file.csss_omni_model

import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.PlasmaML.helios
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer}

@main
def apply(
    start_year: Int = 2011,
    end_year: Int = 2017, 
    divergence_term: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.JensenShannon,
    network_size: Seq[Int] = Seq(100, 80), 
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
    optimization_algo: tf.train.Optimizer = tf.train.Adam(0.01)) = {
    
    val cv_results = (start_year to end_year).map(ty => {

        val result = csss_omni_model(
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
            divergence = divergence_term
        )
        
        csss_omni_model.FTExperiment.clear_cache()
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
  optimization_algo: tf.train.Optimizer = tf.train.Adam(0.01)) = {

    val cv_results = (start_year to end_year).map(ty => {

        val result = csss_omni_model.single_output(
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
            reg = regularization_const
        )

        (ty, result)
    }).toMap

    cv_results

}



//cv_results.toSeq.sortBy(_._1).foreach(kv => {
//    kv._2.results.metrics_test.get.generatePlots()
//})
