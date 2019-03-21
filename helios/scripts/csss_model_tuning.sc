import _root_.ammonite.ops._
import _root_.spire.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe
import _root_.io.github.mandar2812.dynaml.analysis._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.PlasmaML.omni.OMNIData
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{L1Regularization, L2Regularization}
import _root_.io.github.mandar2812.PlasmaML.helios
import breeze.numerics.sigmoid
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer}


@main
def apply(
  start_year: Int                                   = 2011,
  end_year: Int                                     = 2017,
  test_year: Int                                    = 2015,
  sw_threshold: Double                              = 700d,
  divergence_term: helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
  network_size: Seq[Int]                            = Seq(100, 60),
  activation_func: Int => Activation[Double]        = (i: Int) => timelag.utils.getReLUAct[Double](1, i),
  history_fte: Int                                  = 10,
  fte_step: Int                                     = 2,
  crop_latitude: Double                             = 40d,
  log_scale_fte: Boolean                            = false,
  log_scale_omni: Boolean                           = false,
  conv_flag: Boolean                                = false,
  quantity: Int                                     = OMNIData.Quantities.V_SW,
  causal_window: (Int, Int)                         = (48, 56),
  max_iterations: Int                               = 100000,
  max_iterations_tuning: Int                        = 20000,
  num_samples: Int                                  = 20,
  hyper_optimizer: String                           = "gs",
  batch_size: Int                                   = 32,
  optimization_algo: tf.train.Optimizer             = tf.train.AdaDelta(0.01f),
  summary_dir: Path                                 = home/'tmp,
  hyp_opt_iterations: Option[Int]                   = Some(5), 
  reg_type: String                                  = "L2")
: helios.Experiment[Double, fte.ModelRunTuning, fte.FTExperiment.Config] = {


  val num_pred_dims = timelag.utils.get_num_output_dims(
    causal_window._2, mo_flag = true,
    prob_timelags = true, dist_type = "default")

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelag.utils.get_ffnet_properties(-1, num_pred_dims, network_size)

  val output_mapping = timelag.utils.get_output_mapping[Double](
    causal_window._2, mo_flag = true,
    prob_timelags = true, dist_type = "default")

  val filter_depths = Seq(
    Seq(4, 4, 4, 4),
    Seq(2, 2, 2, 2),
    Seq(1, 1, 1, 1)
  )

  val activation = DataPipe[String, Layer[Output[Float], Output[Float]]]((s: String) => tf.learn.ReLU(s, 0.01f))

  //Prediction architecture
  val architecture = if (conv_flag) {
    tf.learn.Cast[Double, Float]("Cast/Input") >>
      dtflearn.inception_unit[Float](
        channels = 1, filter_depths.head,
        activation, use_batch_norm = true)(layer_index = 1) >>
      tf.learn.MaxPool(s"MaxPool_1", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
      dtflearn.inception_unit[Float](
        filter_depths.head.sum, filter_depths(1),
        activation, use_batch_norm = true)(layer_index = 2) >>
      tf.learn.MaxPool(s"MaxPool_2", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
      dtflearn.inception_unit[Float](
        filter_depths(1).sum, filter_depths.last,
        activation, use_batch_norm = true)(layer_index = 3) >>
      tf.learn.MaxPool(s"MaxPool_3", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
      tf.learn.Flatten("FlattenFeatures") >>
      tf.learn.Cast[Float, Double]("Cast/Output") >>
      dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail) >>
      output_mapping
  } else {
    dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail) >>
      output_mapping
  }


  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

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
    "prior_wt"    -> UniformRV(0.5, 1.5),
    "error_wt"    -> UniformRV(0.75, 1.5),
    "temperature" -> UniformRV(0.9, 2.0),
    "specificity" -> UniformRV(1.0, 2.0),
    "reg"         -> UniformRV(math.pow(10d, -5d), math.pow(10d, -3d))
  )

  val hyp_scaling = hyper_prior.map(p =>
    (
      p._1,
      Encoder((x: Double) => (x - p._2.min)/(p._2.max - p._2.min), (u: Double) => u*(p._2.max - p._2.min) + p._2.min)
    )
  )

  val logit = Encoder((x: Double) => math.log(x/(1d - x)), (x: Double) => sigmoid(x))

  val hyp_mapping = Some(
    hyper_parameters.map(
      h => (h, hyp_scaling(h) > logit)
    ).toMap
  )


  val loss_func_generator = (h: Map[String, Double]) => {

    val lossFunc = timelag.utils.get_loss[Double, Double, Double](
      causal_window._2, mo_flag = true,
      prob_timelags = true,
      prior_wt = h("prior_wt"),
      prior_divergence = divergence_term,
      temp = h("temperature"),
      error_wt = h("error_wt"),
      c = h("specificity"))

    val reg =
      if(reg_type == "L2") L2Regularization[Double](layer_scopes, layer_parameter_names, layer_datatypes, layer_shapes, h("reg"))
      else L1Regularization[Double](layer_scopes, layer_parameter_names, layer_datatypes, layer_shapes, h("reg"))
    
    lossFunc >>
      reg >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")
  }

  val fitness_function = DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]]((preds, targets) => {

    val weighted_error = preds._1
      .subtract(targets)
      .square
      .multiply(preds._2)
      .sum(axes = 1)

    val entropy = preds._2
      .multiply(Tensor(-1d).castTo[Double])
      .multiply(tfi.log(preds._2))
      .sum(axes = 1)

    weighted_error + entropy
  })
  
   
  
  fte.exp_cdt_tuning(
    architecture, hyper_parameters,
    loss_func_generator, fitness_function,
    hyper_prior, hyp_mapping = hyp_mapping,
    year_range = start_year to end_year,
    test_year = test_year,
    sw_threshold = sw_threshold,
    quantity = quantity,
    optimizer = optimization_algo,
    miniBatch = batch_size,
    iterations = max_iterations,
    num_samples = num_samples,
    hyper_optimizer = hyper_optimizer,
    iterations_tuning = max_iterations_tuning,
    latitude_limit = crop_latitude,
    deltaTFTE = history_fte,
    fteStep = fte_step,
    conv_flag = conv_flag,
    log_scale_fte = log_scale_fte,
    log_scale_omni = log_scale_omni,
    deltaT = causal_window,
    divergence = divergence_term,
    summary_top_dir = summary_dir,
    hyp_opt_iterations = hyp_opt_iterations)
}