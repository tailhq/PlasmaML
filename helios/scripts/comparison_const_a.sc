import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios.core._
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{L2Regularization, L1Regularization}

@main
def main(
  d: Int                 = 3,
  n: Int                 = 100,
  sliding_window: Int    = 15,
  noise: Double          = 0.5,
  noiserot: Double       = 0.1,
  iterations: Int        = 150000,
  optimizer: Optimizer   = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String = "const_a",
  reg: Double            = 0.01,
  p: Double              = 1.0,
  time_scale: Double     = 1.0,
  corr_sc: Double        = 2.5,
  c_cutoff: Double       = 0.0,
  prior_wt: Double       = 1d) = {

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

  val dataset: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output > compute_time_lag,
    d = d, n = n, noise = noise, noiserot = noiserot,
    alpha = alpha, sliding_window = sliding_window)


  val num_outputs        = sliding_window

  Seq((false, 0.0), (false, corr_sc), (true, prior_wt)).map(config => {

    val (mo_flag, wt) = config

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
        weight_error = wt)
    } else {
      RBFWeightedSWLoss(
        "Loss/RBFWeightedL1", num_outputs,
        kernel_time_scale = time_scale,
        kernel_norm_exponent = p,
        corr_cutoff = c_cutoff,
        prior_scaling = wt,
        batch = 512)
    }

    val loss     = lossFunc >>
      L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    val app_sum_dir_prefix = if(wt > 0) sum_dir_prefix+"_sp" else sum_dir_prefix

    timelag.run_exp(
      dataset, architecture, loss,
      iterations, optimizer, 512, app_sum_dir_prefix,
      mo_flag, loss)
  })
}