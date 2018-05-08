import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios.core._
import _root_.io.github.mandar2812.PlasmaML.utils._

import $file.timelagutils

@main
def main(
  d: Int                        = 3,
  n: Int                        = 100,
  sliding_window: Int           = 15,
  noise: Double                 = 0.5,
  noiserot: Double              = 0.1,
  alpha: Double                 = 0.0,
  num_neurons: Int              = 40,
  num_hidden_layers: Int        = 1,
  iterations: Int               = 150000,
  miniBatch: Int                = 32,
  optimizer: Optimizer          = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String        = "const_v",
  reg: Double                   = 0.01,
  p: Double                     = 1.0,
  time_scale: Double            = 1.0,
  corr_sc: Double               = 2.5,
  c_cutoff: Double              = 0.0,
  prior_wt: Double              = 1d,
  prior_type: String            = "Hellinger",
  mo_flag: Boolean              = true,
  prob_timelags: Boolean        = true,
  timelag_pred_strategy: String = "mode") = {

  //Output computation
  val beta = 100f


  //Time Lag Computation
  // distance/velocity
  val distance = beta*10
  val compute_output: DataPipe[Tensor, (Float, Float)] = DataPipe(
    (v: Tensor) => {

      val out = v.square.sum().scalar.asInstanceOf[Float]*beta + 40f

      (distance/out, out + scala.util.Random.nextGaussian().toFloat)
    })

  val num_pred_dims =
    if(!mo_flag) 2
    else if(mo_flag && !prob_timelags) sliding_window + 1
    else if(!mo_flag && prob_timelags) sliding_window + 1
    else 2*sliding_window

  val num_outputs           = sliding_window
  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelagutils.get_ffnet_properties(d, num_pred_dims, num_neurons, num_hidden_layers)

  //Prediction architecture
  val architecture = dtflearn.feedforward_stack(
    (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
    net_layer_sizes.tail)

  val lossFunc = if (!mo_flag) {

    if (!prob_timelags) {
      RBFWeightedSWLoss(
        "Loss/RBFWeightedL1", num_outputs,
        kernel_time_scale = time_scale,
        kernel_norm_exponent = p,
        corr_cutoff = c_cutoff,
        prior_scaling = corr_sc,
        batch = 512)
    } else {
      WeightedTimeSeriesLossSO(
        "Loss/ProbWeightedTS",
        num_outputs,
        prior_wt = prior_wt,
        temperature = 0.75)
    }

  } else if(mo_flag && !prob_timelags) {

    MOGrangerLoss(
      "Loss/MOGranger", num_outputs,
      error_exponent = p,
      weight_error = prior_wt)

  } else {

    WeightedTimeSeriesLoss(
      "Loss/ProbWeightedTS",
      num_outputs,
      prior_wt = prior_wt,
      temperature = 0.75)

  }

  val loss = lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val dataset: timelagutils.TLDATA = timelagutils.generate_data(
    d, n, sliding_window,
    noise, noiserot, alpha,
    compute_output)

  timelagutils.run_exp(
    dataset, iterations, optimizer,
    miniBatch, sum_dir_prefix,
    mo_flag, prob_timelags,
    timelag_pred_strategy,
    architecture, loss)
}