import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{L2Regularization, L1Regularization}
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.ammonite.ops._
import org.platanios.tensorflow.api.learn.layers.Activation

@main
def main(
  d: Int                             = 3,
  confounding: Double                = 0d,
  size_training: Int                 = 100,
  size_test: Int                     = 50,
  sliding_window: Int                = 15,
  noise: Double                      = 0.5,
  noiserot: Double                   = 0.1,
  alpha: Double                      = 0.0,
  train_test_separate: Boolean       = false,
  num_neurons: Seq[Int]              = Seq(40),
  activation_func: Int => Activation = timelag.utils.getReLUAct(1),
  iterations: Int                    = 150000,
  miniBatch: Int                     = 32,
  optimizer: Optimizer               = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String             = "const_v",
  reg: Double                        = 0.01,
  p: Double                          = 1.0,
  time_scale: Double                 = 1.0,
  corr_sc: Double                    = 2.5,
  c_cutoff: Double                   = 0.0,
  prior_wt: Double                   = 1d,
  c: Double                          = 1d,
  prior_type: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
  temp: Double                       = 1.0,
  error_wt: Double                   = 1.0,
  mo_flag: Boolean                   = true,
  prob_timelags: Boolean             = true,
  dist_type: String                  = "default",
  timelag_pred_strategy: String      = "mode",
  summaries_top_dir: Path            = home/'tmp): timelag.ExperimentResult[timelag.JointModelRun] = {

  //Output computation
  val beta = 100f


  //Time Lag Computation
  // distance/velocity
  val distance = beta*10

  val compute_v = DataPipe[Tensor, Float]((v: Tensor) => v.square.mean().scalar.asInstanceOf[Float]*beta/d + 40f)

  val compute_output: DataPipe[Tensor, (Float, Float)] = DataPipe(
    (x: Tensor) => {

      val out = compute_v(x)

      val noisy_output = out + scala.util.Random.nextGaussian().toFloat

      (distance/noisy_output, noisy_output)
    })

  val num_pred_dims = timelag.utils.get_num_output_dims(sliding_window, mo_flag, prob_timelags, dist_type)

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelag.utils.get_ffnet_properties(d, num_pred_dims, num_neurons, "FLOAT32")

  val output_mapping = timelag.utils.get_output_mapping(
    sliding_window, mo_flag,
    prob_timelags, dist_type,
    time_scale)

  //Prediction architecture
  val architecture = dtflearn.feedforward_stack(
    activation_func, FLOAT32)(
    net_layer_sizes.tail) >>
    output_mapping


  val lossFunc = timelag.utils.get_loss(
    sliding_window, mo_flag,
    prob_timelags, p, time_scale,
    corr_sc, c_cutoff,
    prior_wt, prior_type,
    temp, error_wt, c)

  val loss = lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val dataset: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output, sliding_window,
    d, size_training, noiserot,
    alpha, noise)

  val experiment_result = if(train_test_separate) {
    val dataset_test: timelag.utils.TLDATA = timelag.utils.generate_data(
      compute_output, sliding_window,
      d, size_test, noiserot,
      alpha, noise)

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
      dataset,
      architecture, loss,
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
      output_mapping = Some(compute_v)
    )
  )
}


def stage_wise(
  d: Int                             = 3,
  confounding: Double                = 0d,
  size_training: Int                 = 100,
  size_test: Int                     = 50,
  sliding_window: Int                = 15,
  noise: Double                      = 0.5,
  noiserot: Double                   = 0.1,
  alpha: Double                      = 0.0,
  num_neurons_i: Seq[Int]            = Seq(40),
  num_neurons_ii: Seq[Int]           = Seq(40),
  activation_func_i: Int => Activation = timelag.utils.getReLUAct(1),
  activation_func_ii: Int => Activation = timelag.utils.getReLUAct(10),
  iterations: Int                    = 150000,
  miniBatch: Int                     = 32,
  optimizer: Optimizer               = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String             = "const_v",
  reg_i: Double                      = 0.01,
  reg_ii: Double                     = 0.001,
  prior_wt: Double                   = 1d,
  c: Double                          = 1d,
  temp: Double                       = 1d,
  prior_type: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
  error_wt: Double                   = 1.0,
  mo_flag: Boolean                   = true,
  prob_timelags: Boolean             = true,
  dist_type: String                  = "default",
  timelag_pred_strategy: String      = "mode",
  summaries_top_dir: Path            = home/'tmp): timelag.ExperimentResult[timelag.StageWiseModelRun] = {

  //Output computation
  val beta = 100f
  //Time Lag Computation
  // distance/velocity
  val distance = beta*10
  val compute_output: DataPipe[Tensor, (Float, Float)] = DataPipe(
    (v: Tensor) => {

      val out = v.square.mean().scalar.asInstanceOf[Float]*beta/d + 40f

      (distance/out, out + scala.util.Random.nextGaussian().toFloat)
    })


  val dataset: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output, sliding_window,
    d, size_training, noiserot,
    alpha, noise)

  val dataset_test: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output, sliding_window,
    d, size_test, noiserot,
    alpha, noise)

  val (net_layer_sizes_i, layer_shapes_i, layer_parameter_names_i, layer_datatypes_i) =
    timelag.utils.get_ffnet_properties(
      d, sliding_window,
      num_neurons_i,
      dType = "FLOAT32")

  val (net_layer_sizes_ii, layer_shapes_ii, layer_parameter_names_ii, layer_datatypes_ii) =
    timelag.utils.get_ffnet_properties(
      d, sliding_window, num_neurons_ii,
      dType = "FLOAT32",
      net_layer_sizes_i.length - 1)

  //Prediction architecture
  val architecture_i = dtflearn.feedforward_stack(
    activation_func_i, FLOAT32)(
    net_layer_sizes_i.tail)

  val architecture_ii = dtflearn.feedforward_stack(
    activation_func_ii, FLOAT32)(
    net_layer_sizes_ii.tail,
    net_layer_sizes_i.length - 1) >>
    helios.learn.cdt_ii.output_mapping("Prob")


  val loss_i  = helios.learn.cdt_i("OutputLoss", sliding_window) >>
    L2Regularization(layer_parameter_names_i, layer_datatypes_i, layer_shapes_i, reg_i) >>
    tf.learn.ScalarSummary("Loss", "OutputLoss")


  val loss_ii = helios.learn.cdt_ii(
    "TimeLagLoss", sliding_window, prior_wt, error_wt,
    temperature = temp, specificity = c,
    divergence = prior_type) >>
    L2Regularization(layer_parameter_names_ii, layer_datatypes_ii, layer_shapes_ii, reg_ii) >>
    tf.learn.ScalarSummary("Loss", "TimeLagLoss")

  val experiment_result = timelag.run_exp_stage_wise(
    (dataset, dataset_test),
    architecture_i, architecture_ii, loss_i, loss_ii,
    iterations, optimizer, miniBatch, sum_dir_prefix,
    mo_flag, prob_timelags, timelag_pred_strategy,
    summaries_top_dir, confounding_factor = confounding
  )

  experiment_result.copy(
    config = experiment_result.config.copy(
      output_mapping = Some(compute_output > DataPipe[(Float, Float), Float](_._1))
    )
  )
}