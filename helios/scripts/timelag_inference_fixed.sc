import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{L2Regularization, L1Regularization}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.ammonite.ops._
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag

@main
def main(
  fixed_lag: Float              = 3f,
  d: Int                        = 3,
  confounding: Double           = 0d,
  size_training: Int            = 100,
  size_test: Int                = 50,
  sliding_window: Int           = 15,
  noise: Double                 = 0.5,
  noiserot: Double              = 0.1,
  alpha: Double                 = 0.0,
  train_test_separate: Boolean  = false,
  num_neurons: Seq[Int]         = Seq(40),
  iterations: Int               = 150000,
  miniBatch: Int                = 32,
  optimizer: Optimizer          = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String        = "const_lag",
  reg: Double                   = 0.01,
  p: Double                     = 1.0,
  time_scale: Double            = 1.0,
  corr_sc: Double               = 2.5,
  c_cutoff: Double              = 0.0,
  prior_wt: Double              = 1d,
  c: Double                     = 1d,
  prior_type: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
  target_dist: helios.learn.cdt_loss.TargetDistribution = helios.learn.cdt_loss.Boltzmann,
  temp: Double                  = 1.0,
  error_wt: Double              = 1.0,
  mo_flag: Boolean              = true,
  prob_timelags: Boolean        = true,
  dist_type: String             = "default",
  timelag_pred_strategy: String = "mode",
  summaries_top_dir: Path       = home/'tmp): timelag.ExperimentResult[timelag.JointModelRun] = {

  //Output computation
  val beta = 100f
  //Time Lag Computation

  val compute_v = DataPipe[Tensor, Float]((v: Tensor) => v.square.mean().scalar.asInstanceOf[Float]*beta/d)

  val compute_output: DataPipe[Tensor, (Float, Float)] = DataPipe(
    (x: Tensor) => {

      val out = compute_v(x)

      (fixed_lag, out + scala.util.Random.nextGaussian().toFloat)
    })

  val num_pred_dims = timelag.utils.get_num_output_dims(sliding_window, mo_flag, prob_timelags, dist_type)

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelag.utils.get_ffnet_properties(d, num_pred_dims, num_neurons)

  val output_mapping = timelag.utils.get_output_mapping(sliding_window, mo_flag, prob_timelags, dist_type, time_scale)

  //Prediction architecture
  val architecture = dtflearn.feedforward_stack(
    (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
    net_layer_sizes.tail) >> output_mapping


  val lossFunc = timelag.utils.get_loss(
    sliding_window, mo_flag,
    prob_timelags, p, time_scale,
    corr_sc, c_cutoff,
    prior_wt, prior_type, target_dist,
    temp, error_wt, c)

  val loss     = lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val dataset: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output, sliding_window,
    d, size_training, noiserot,
    alpha, noise)

  val experiment_result = if(train_test_separate) {
    val dataset_test: timelag.utils.TLDATA = timelag.utils.generate_data(
      compute_output, sliding_window, d,
      size_test, noiserot, alpha,
      noise)

    timelag.run_exp_joint(
      (dataset, dataset_test),
      architecture, loss,
      iterations, optimizer,
      miniBatch, sum_dir_prefix,
      mo_flag, prob_timelags,
      timelag_pred_strategy,
      summaries_top_dir,
      confounding_factor = confounding)

  } else {

    timelag.run_exp(
      dataset, architecture, loss,
      iterations, optimizer,
      miniBatch,
      sum_dir_prefix,
      mo_flag, prob_timelags,
      timelag_pred_strategy,
      summaries_top_dir,
      confounding_factor = confounding)
  }

  experiment_result.copy(
    config = experiment_result.config.copy(
      output_mapping = Some(compute_v))
    )

}