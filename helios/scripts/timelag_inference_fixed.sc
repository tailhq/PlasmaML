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
  fixed_lag: Float       = 3f,
  d: Int                 = 3,
  n: Int                 = 100,
  sliding_window: Int    = 15,
  noise: Double          = 0.5,
  noiserot: Double       = 0.1,
  iterations: Int        = 150000,
  optimizer: Optimizer   = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String = "const_lag",
  reg: Double            = 0.01,
  p: Double              = 1.0,
  time_scale: Double     = 1.0,
  corr_sc: Double        = 2.5,
  c_cutoff: Double       = 0.0,
  prior_wt: Double       = 1d,
  mo_flag: Boolean       = false,
  prob_timelags: Boolean = false) = {

  //Output computation
  val alpha = 100f
  //Time Lag Computation
  val compute_output: DataPipe[Tensor, (Float, Float)] = DataPipe(
    (v: Tensor) => {

      val out = v.square.sum().scalar.asInstanceOf[Float]*alpha

      (fixed_lag, out + scala.util.Random.nextGaussian().toFloat*noise.toFloat)
    })

  val num_outputs           = sliding_window

  val num_pred_dims =
    if(!mo_flag) 2
    else if(mo_flag && !prob_timelags) sliding_window + 1
    else 2*sliding_window

  val net_layer_sizes       = Seq(d, 20, num_pred_dims)

  val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))

  val layer_parameter_names = (1 to net_layer_sizes.tail.length).map(s => "Linear_"+s+"/Weights")

  val layer_datatypes       = Seq.fill(net_layer_sizes.tail.length)("FLOAT64")

  //Prediction architecture
  val architecture          = dtflearn.feedforward_stack(
    (i: Int) => tf.learn.Sigmoid("Act_"+i), FLOAT64)(
    net_layer_sizes.tail)

  val lossFunc = if (!mo_flag){

    RBFWeightedSWLoss(
      "Loss/RBFWeightedL1", num_outputs,
      kernel_time_scale = time_scale,
      kernel_norm_exponent = p,
      corr_cutoff = c_cutoff,
      prior_scaling = corr_sc,
      batch = 512)

  } else if(mo_flag && !prob_timelags){
    MOGrangerLoss(
      "Loss/MOGranger", num_outputs,
      error_exponent = p,
      weight_error = prior_wt)
  } else {
    WeightedTimeSeriesLoss("Loss/ProbWeightedTS", num_outputs)
  }

  val loss     = lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val dataset: timelagutils.TLDATA = timelagutils.generate_data(
    d, n, sliding_window,
    noise, noiserot,
    compute_output)

  timelagutils.run_exp(
    dataset,
    iterations, optimizer, 512, sum_dir_prefix,
    mo_flag, prob_timelags,
    architecture, loss)
}