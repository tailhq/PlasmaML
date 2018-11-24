import $file.csss_omni_model

import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.PlasmaML.helios

@main
def apply(
    start_year: Int = 2011: Int, 
    end_year: Int = 2017, 
    divergence_term: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
    network_size: Seq[Int] = Seq(100, 80), 
    history_fte: Int = 5, 
    crop_latitude: Double = 40d, 
    log_scale_fte: Boolean = false, 
    causal_window: (Int, Int) = (48, 72),
    max_iterations: Int = 100000, 
    batch_size: Int = 32, 
    regularization_const: Double = 0.001, 
    optimization_algo: tf.train.Optimizer = tf.train.AdaDelta(0.01)) = {
    
    val cv_results = (start_year to end_year).map(ty => {

        val result = csss_omni_model(
            year_range = start_year to end_year, 
            test_year = ty, 
            optimizer = optimization_algo,
            num_neurons = network_size, 
            miniBatch = batch_size, 
            iterations = max_iterations, 
            latitude_limit = crop_latitude, 
            deltaTFTE = history_fte, 
            log_scale_fte = log_scale_fte,
            deltaT = causal_window, 
            reg = regularization_const, 
            divergence = divergence_term
        )

        (ty, result)
    }).toMap

  cv_results

}



//cv_results.toSeq.sortBy(_._1).foreach(kv => {
//    kv._2.results.metrics_test.get.generatePlots()
//})
