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
  d: Int                 = 3,
  n: Int                 = 100,
  sliding_window: Int    = 15,
  noise: Double          = 0.5,
  noiserot: Double       = 0.1,
  iterations: Int        = 150000,
  optimizer: Optimizer   = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String = "",
  reg: Double            = 0.01,
  p: Double              = 1.0,
  time_scale: Double     = 1.0,
  corr_sc: Double        = 2.5,
  c_cutoff: Double       = 0.0,
  prior_wt: Double       = 1d,
  mo_flag: Boolean       = false) = {

  //Output computation
  val alpha = 100f
  val compute_output = DataPipe(
    (v: Tensor) =>
      (
        v.square.sum().scalar.asInstanceOf[Float]*alpha,
        alpha*0.1f
      )
  )

  //Time Lag Computation
  // 1/2*a*t^2 + u*t - s = 0
  // t = (-u + sqrt(u*u + 2as))/a
  val distance = alpha*20

  val compute_time_lag = DataPipe((va: (Float, Float)) => {
    val (v, a) = va
    val dt = (-v + math.sqrt(v*v + 2*a*distance).toFloat)/a
    val vf = math.sqrt(v*v + 2f*a*distance).toFloat
    (dt, vf + scala.util.Random.nextGaussian().toFloat*noise.toFloat)
  })

  val num_outputs        = sliding_window

  val num_pred_dims = if(mo_flag) sliding_window + 1 else 2

  val net_layer_sizes = Seq(d, 20, 15, num_pred_dims)

  val layer_shapes = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))

  val layer_parameter_names = (1 to net_layer_sizes.tail.length).map(s => "Linear_"+s+"/Weights")

  val layer_datatypes = Seq.fill(net_layer_sizes.tail.length)("FLOAT64")

  //Prediction architecture
  val architecture = dtflearn.feedforward_stack(
    (i: Int) => tf.learn.Sigmoid("Act_"+i), FLOAT64)(
    net_layer_sizes.tail)

  val lossFunc = if (mo_flag) {
    MOGrangerLoss(
      "Loss/MOGranger", num_outputs,
      error_exponent = p,
      weight_error = prior_wt)
  } else {
    RBFWeightedSWLoss(
      "Loss/RBFWeightedL1", num_outputs,
      kernel_time_scale = time_scale,
      kernel_norm_exponent = p,
      corr_cutoff = c_cutoff,
      prior_scaling = corr_sc,
      prior_weight = prior_wt,
      batch = 512)
  }

  val loss     = lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  timelagutils.run_exp(
    d, n, sliding_window, noise, noiserot,
    compute_output > compute_time_lag,
    iterations, optimizer, 512, sum_dir_prefix,
    mo_flag, architecture, loss)
}