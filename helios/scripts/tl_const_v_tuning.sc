import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.data.DataSet
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe
import _root_.io.github.mandar2812.dynaml.analysis._
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.models.TunableTFModel
import _root_.spire.implicits._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.utils._
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.ammonite.ops._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Activation

@main
def main(
  d: Int                             = 3,
  n: Int                             = 100,
  sliding_window: Int                = 15,
  noise: Double                      = 0.5,
  noiserot: Double                   = 0.1,
  alpha: Double                      = 0.0,
  train_test_separate: Boolean       = false,
  num_neurons: Seq[Int]              = Seq(40),
  activation_func: Int => Activation = timelag.utils.getReLUAct(1),
  iterations: Int                    = 150000,
  iterations_tuning: Int             = 20000,
  num_samples: Int                   = 20,
  miniBatch: Int                     = 32,
  optimizer: Optimizer               = tf.train.AdaDelta(0.01),
  sum_dir_prefix: String             = "const_v",
  prior_type: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
  dist_type: String                  = "default",
  timelag_pred_strategy: String      = "mode",
  summaries_top_dir: Path            = home/'tmp): timelag.ExperimentResult[timelag.TunedModelRun] = {

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

  val num_pred_dims = timelag.utils.get_num_output_dims(sliding_window, mo_flag, prob_timelags, dist_type)

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelag.utils.get_ffnet_properties(d, num_pred_dims, num_neurons, "FLOAT32")

  val output_mapping = timelag.utils.get_output_mapping(
    sliding_window, mo_flag,
    prob_timelags, dist_type)

  //Prediction architecture
  val architecture = dtflearn.feedforward_stack(
    activation_func, FLOAT32)(
    net_layer_sizes.tail) >>
    output_mapping


  implicit val detImpl = DynaMLPipe.identityPipe[Double]

  val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
    DataPipe((x: Double) => math.exp(x)),
    DifferentiableMap(
      (x: Double) => math.log(x),
      (x: Double) => 1.0/x)
  )

  val h10: PushforwardMap[Double, Double, Double] = PushforwardMap(
    DataPipe((x: Double) => math.pow(10d, x)),
    DifferentiableMap(
      (x: Double) => math.log10(x),
      (x: Double) => 1.0/(x*math.log(10d)))
  )

  val g1 = GaussianRV(0.0, 0.75)

  val g2 = GaussianRV(0.2, 0.75)

  val lg_p = h -> g1
  val lg_e = h -> g2

  val lu_reg = h10 -> UniformRV(-4d, -2.5d)

  val hyper_parameters = List(
    "prior_wt",
    "error_wt",
    "temperature",
    "specificity",
    "reg"
  )

  val hyper_prior = Map(
    "prior_wt"    -> lg_p,
    "error_wt"    -> lg_e,
    "temperature" -> UniformRV(0.9, 2.0),
    "specificity" -> UniformRV(1.0, 2.0),
    "reg"         -> lu_reg
  )


  val loss_func_generator = (h: Map[String, Double]) => {

    val lossFunc = timelag.utils.get_loss(
      sliding_window, mo_flag,
      prob_timelags,
      prior_wt = h("prior_wt"),
      prior_divergence = prior_type,
      temp = h("temperature"),
      error_wt = h("error_wt"),
      c = h("specificity"))

    lossFunc >>
      L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, h("reg")) >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")
  }

  val fitness_function = DataPipe[DataSet[((Tensor, Tensor), Tensor)], Double](validation_coll => {

    //collect tensors into one big tensor.
    val get_preds: (((Tensor, Tensor), Tensor)) => Tensor = _._1._1
    val get_lag_probs: (((Tensor, Tensor), Tensor)) => Tensor = _._1._2
    val get_targets: (((Tensor, Tensor), Tensor)) => Tensor = _._2



    val preds = tfi.stack(validation_coll.map(get_preds).data.toSeq, axis = -1)
    val prob_lags = tfi.stack(validation_coll.map(get_lag_probs).data.toSeq, axis = -1)
    val targets = tfi.stack(validation_coll.map(get_targets).data.toSeq, axis = -1)


    preds
      .subtract(targets)
      .square
      .multiply(prob_lags)
      .sum(axes = 1)
      .mean()
      .scalar
      .asInstanceOf[Double]
  })

  val dataset: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output, d, n, noise, noiserot,
    alpha, sliding_window)

  val dataset_test: timelag.utils.TLDATA = timelag.utils.generate_data(
    compute_output, d, n, noise, noiserot,
    alpha, sliding_window)



  timelag.run_exp_hyp(
    (dataset, dataset_test),
    architecture, hyper_parameters,
    loss_func_generator,
    fitness_function, hyper_prior,
    iterations, iterations_tuning,
    num_samples, optimizer,
    miniBatch, sum_dir_prefix,
    mo_flag, prob_timelags,
    timelag_pred_strategy,
    summaries_top_dir)

}