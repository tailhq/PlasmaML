import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.ammonite.ops._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Activation

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
  activation_func: Int => Activation[Double]                 = timelag.utils.getReLUAct[Double](1, _),
  iterations: Int                                            = 150000,
  iterations_tuning: Int                                     = 20000,
  miniBatch: Int                                             = 32,
  optimizer: Optimizer                                       = tf.train.AdaDelta(0.01f),
  sum_dir_prefix: String                                     = "softplus",
  prior_type: Seq[helios.learn.cdt_loss.Divergence]          = Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_prob: Seq[helios.learn.cdt_loss.TargetDistribution] = Seq(helios.learn.cdt_loss.Boltzmann),
  dist_type: String                                          = "default",
  timelag_pred_strategy: String                              = "mode",
  summaries_top_dir: Path                                    = home/'tmp,
  num_samples: Int                                           = 20,
  hyper_optimizer: String                                    = "gs",
  hyp_opt_iterations: Option[Int]                            = Some(5),
  epochFlag: Boolean                                         = false,
  regularization_types: Seq[String]                          = Seq("L2"))
: Seq[timelag.ExperimentResult[Double, Double, timelag.TunedModelRun[Double, Double]]] = {

  //Output computation
  val beta = 100f

  val compute_v = DataPipe[Tensor[Double], Float]((v: Tensor[Double]) => v.square.mean().scalar.asInstanceOf[Float]*beta/d + 40f)

  val compute_output: DataPipe[Tensor[Double], (Float, Float)] = DataPipe(
    (x: Tensor[Double]) => {

      val out = compute_v(x)

      (math.log(1 + math.exp(out/20.0)).toFloat, out + scala.util.Random.nextGaussian().toFloat)
    })

  val experiment_results = run_model_tuning_pdt(
    compute_output,
    d, confounding, size_training, size_test, sliding_window, noise, noiserot,
    alpha, train_test_separate, num_neurons,
    activation_func, iterations, iterations_tuning,
    miniBatch, optimizer, sum_dir_prefix,
    prior_type, target_prob, dist_type,
    timelag_pred_strategy, summaries_top_dir, num_samples,
    hyper_optimizer, hyp_opt_iterations, epochFlag,
    regularization_types
  )

  experiment_results.map(experiment_result =>
    experiment_result.copy[Double, Double, timelag.TunedModelRun[Double, Double]](
      config = experiment_result.config.copy[Double](output_mapping = Some(compute_v))
  ))
}