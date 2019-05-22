import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.ammonite.ops._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Activation
import org.platanios.tensorflow.api.learn.layers.Layer

import $file.run_model_tuning_pdt

@main
def main(
  d: Int                                                     = 3,
  confounding: Seq[Double]                                   = Seq(0d, 0.25, 0.5, 0.75),
  size_training: Int                                         = 100,
  size_test: Int                                             = 50,
  sliding_window: Int                                        = 15,
  noise: Double                                              = 0.5,
  noiserot: Double                                           = 0.1,
  alpha: Double                                              = 0.0,
  train_test_separate: Boolean                               = false,
  num_neurons: Seq[Int]                                      = Seq(40),
  activation_func: Int => Layer[Output[Double], Output[Double]] 
    = timelag.utils.getReLUAct[Double](1, _),
  iterations: Int                                            = 150000,
  iterations_tuning: Int                                     = 20000,
  pdt_iterations: Int                                        = 2,
  pdt_iterations_tuning: Int                                 = 4,
  miniBatch: Int                                             = 32,
  optimizer: Optimizer                                       = tf.train.AdaDelta(0.01f),
  sum_dir_prefix: String                                     = "const_a",
  summaries_top_dir: Path                                    = home/'tmp,
  num_samples: Int                                           = 20,
  hyper_optimizer: String                                    = "gs",
  hyp_opt_iterations: Option[Int]                            = Some(5),
  epochFlag: Boolean                                         = false,
  regularization_types: Seq[String]                          = Seq("L2"),
  checkpointing_freq: Int                                    = 4)
: Seq[timelag.ExperimentResult[Double, Double, timelag.TunedModelRun[Double, Double]]] = {

  //Output computation
  val beta = 100f
  val compute_output = DataPipe(
    (v: Tensor[Double]) =>
      (
        v.square.mean().scalar.asInstanceOf[Float]*beta*1f/d + 100,
        beta*0.1f
      )
  )

  //Time Lag Computation
  // 1/2*a*t^2 + u*t - s = 0
  // t = (-u + sqrt(u*u + 2as))/a
  val distance = beta*10

  val compute_time_lag = DataPipe((va: (Float, Float)) => {
    val (v, a) = va
    val dt = (math.sqrt(v*v + 2*a*distance).toFloat - v)/a
    val vf = math.sqrt(v*v + 2f*a*distance).toFloat
    (dt, vf)
  })

  val add_noise = DataPipe[(Float, Float), (Float, Float)](
    p => (p._1, p._2 + scala.util.Random.nextGaussian().toFloat))

  val experiment_results = run_model_tuning_pdt(
    compute_output > compute_time_lag > add_noise,
    d, confounding, size_training, size_test, sliding_window, noise, noiserot,
    alpha, train_test_separate, num_neurons, 
    activation_func, iterations, iterations_tuning, 
    pdt_iterations, pdt_iterations_tuning,
    miniBatch, optimizer, sum_dir_prefix,
    summaries_top_dir, num_samples,
    hyper_optimizer, hyp_opt_iterations, epochFlag,
    regularization_types, checkpointing_freq
  )

  experiment_results.map(experiment_result =>
    experiment_result.copy[Double, Double, timelag.TunedModelRun[Double, Double]](
      config = experiment_result.config.copy[Double](
        output_mapping = Some(compute_output > compute_time_lag > DataPipe[(Float, Float), Float](_._2))
    ))
  )
}