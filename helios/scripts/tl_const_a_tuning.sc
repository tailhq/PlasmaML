import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.ammonite.ops._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Activation

import $file.run_model_tuning_cdt

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
  activation_func: Int => Activation                         = timelag.utils.getReLUAct(1),
  iterations: Int                                            = 150000,
  iterations_tuning: Int                                     = 20000,
  miniBatch: Int                                             = 32,
  optimizer: Optimizer                                       = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String                                     = "const_a",
  prior_type: Seq[helios.learn.cdt_loss.Divergence]          = Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_prob: Seq[helios.learn.cdt_loss.TargetDistribution] = Seq(helios.learn.cdt_loss.Boltzmann),
  dist_type: String                                          = "default",
  timelag_pred_strategy: String                              = "mode",
  summaries_top_dir: Path                                    = home/'tmp,
  num_samples: Int                                           = 20,
  hyper_optimizer: String                                    = "gs",
  hyp_opt_iterations: Option[Int]                            = Some(5),
  epochFlag: Boolean                                         = false,
  regularization_type: String                                = "L2")
: Seq[timelag.ExperimentResult[timelag.TunedModelRun]] = {

  //Output computation
  val beta = 100f
  val compute_output = DataPipe(
    (v: Tensor) =>
      (
        v.square.mean().scalar.asInstanceOf[Float]*beta*0.5f/d,
        beta*0.05f
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
    (dt, vf + scala.util.Random.nextGaussian().toFloat)
  })


  val experiment_results = run_model_tuning_cdt(
    compute_output > compute_time_lag,
    d, confounding, size_training, size_test, sliding_window, noise, noiserot,
    alpha, train_test_separate, num_neurons, 
    activation_func, iterations, iterations_tuning, 
    miniBatch, optimizer, sum_dir_prefix,
    prior_type, target_prob, dist_type,
    timelag_pred_strategy, summaries_top_dir, num_samples,
    hyper_optimizer, hyp_opt_iterations, epochFlag
  )

  experiment_results.map(experiment_result => experiment_result.copy(
    config = experiment_result.config.copy(
      output_mapping = Some(compute_output > DataPipe[(Float, Float), Float](_._1))
    ))
  )
}