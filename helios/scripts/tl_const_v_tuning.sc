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
  d: Int                                       = 3,
  n: Int                                       = 100,
  sliding_window: Int                          = 15,
  noise: Double                                = 0.5,
  noiserot: Double                             = 0.1,
  alpha: Double                                = 0.0,
  train_test_separate: Boolean                 = false,
  num_neurons: Seq[Int]                        = Seq(40),
  activation_func: Int => Activation           = timelag.utils.getReLUAct(1),
  iterations: Int                              = 150000,
  iterations_tuning: Int                       = 20000,
  num_samples: Int                             = 20,
  miniBatch: Int                               = 32,
  optimizer: Optimizer                         = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String                       = "const_v",
  prior_type: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
  dist_type: String                            = "default",
  timelag_pred_strategy: String                = "mode",
  summaries_top_dir: Path                      = home/'tmp): timelag.ExperimentResult[timelag.TunedModelRun] = {

  //Output computation
  val beta = 100f
  val mo_flag = true
  val prob_timelags = true


  //Time Lag Computation
  // distance/velocity
  val distance = beta*10
  val compute_output: DataPipe[Tensor, (Float, Float)] = DataPipe(
    (v: Tensor) => {

      val out = v.square.mean().scalar.asInstanceOf[Float]*beta/d + 40f

      val noisy_output = out + scala.util.Random.nextGaussian().toFloat

      (distance/noisy_output, noisy_output)
  })

  run_model_tuning_cdt(
    compute_output,
    d, n, sliding_window, noise, noiserot, 
    alpha, train_test_separate, num_neurons, 
    activation_func, iterations, iterations_tuning, 
    num_samples, miniBatch, optimizer, sum_dir_prefix, 
    prior_type, dist_type, timelag_pred_strategy, 
    summaries_top_dir
  )
}