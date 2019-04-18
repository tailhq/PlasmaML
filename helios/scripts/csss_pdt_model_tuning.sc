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
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{
  L1Regularization,
  L2Regularization
}
import _root_.io.github.mandar2812.PlasmaML.helios
import breeze.numerics.sigmoid
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer}

@main
def apply(
  start_year: Int = 2011,
  end_year: Int = 2017,
  test_year: Int = 2015,
  sw_threshold: Double = 700d,
  network_size: Seq[Int] = Seq(100, 60),
  activation_func: Int => Layer[Output[Double], Output[Double]] = (i: Int) =>
    timelag.utils.getReLUAct[Double](1, i),
  history_fte: Int = 10,
  fte_step: Int = 2,
  crop_latitude: Double = 40d,
  fraction_pca: Double = 0.8,
  log_scale_fte: Boolean = false,
  log_scale_omni: Boolean = false,
  conv_flag: Boolean = false,
  quantity: Int = OMNIData.Quantities.V_SW,
  causal_window: (Int, Int) = (48, 56),
  max_iterations: Int = 100000,
  max_iterations_tuning: Int = 20000,
  pdt_iterations: Int = 4,
  num_samples: Int = 4,
  hyper_optimizer: String = "gs",
  batch_size: Int = 32,
  optimization_algo: tf.train.Optimizer = tf.train.AdaDelta(0.01f),
  summary_dir: Path = home / 'tmp,
  hyp_opt_iterations: Option[Int] = Some(5),
  get_training_preds: Boolean = false,
  reg_type: String = "L2",
  existing_exp: Option[Path] = None,
  checkpointing_freq: Int = 1
): helios.Experiment[Double, fte.ModelRunTuning, fte.data.FteOmniConfig] = {

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      -1,
      network_size.last,
      network_size.take(network_size.length - 1),
      "FLOAT64"
    )

  val output_mapping = helios.learn.cdt_loss.output_mapping[Double](
    "PDTNetwork",
    causal_window._2
  )

  val hyper_parameters = List(
    "sigma_sq",
    "alpha",
    "reg"
  )

  val persistent_hyper_parameters = List("reg")

  val hyper_prior = Map(
    "sigma_sq" -> UniformRV(1e-5, 5d),
    "alpha"    -> UniformRV(0.75d, 2d),
    "reg"      -> UniformRV(-5d, -3d)
  )

  val params_enc = Encoder(
    identityPipe[Map[String, Double]],
    identityPipe[Map[String, Double]]
  )

  val filter_depths = Seq(
    Seq(4, 4, 4, 4),
    Seq(2, 2, 2, 2),
    Seq(1, 1, 1, 1)
  )

  val activation = DataPipe[String, Layer[Output[Float], Output[Float]]](
    (s: String) => tf.learn.ReLU(s, 0.01f)
  )

  //Prediction architecture
  val architecture = if (conv_flag) {
    tf.learn.Cast[Double, Float]("Cast/Input") >>
      dtflearn.inception_unit[Float](
        channels = 1,
        filter_depths.head,
        activation,
        use_batch_norm = true
      )(layer_index = 1) >>
      tf.learn.MaxPool(s"MaxPool_1", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
      dtflearn.inception_unit[Float](
        filter_depths.head.sum,
        filter_depths(1),
        activation,
        use_batch_norm = true
      )(layer_index = 2) >>
      tf.learn.MaxPool(s"MaxPool_2", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
      dtflearn.inception_unit[Float](
        filter_depths(1).sum,
        filter_depths.last,
        activation,
        use_batch_norm = true
      )(layer_index = 3) >>
      tf.learn.MaxPool(s"MaxPool_3", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
      tf.learn.Flatten("FlattenFeatures") >>
      tf.learn.Cast[Float, Double]("Cast/Output") >>
      dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail) >>
      output_mapping
  } else {
    dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail) >>
      activation_func(net_layer_sizes.tail.length) >>
      output_mapping
  }

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

  val output_scope = scope("Outputs")

  implicit val detImpl = DynaMLPipe.identityPipe[Double]

  val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
    DataPipe((x: Double) => math.exp(x)),
    DifferentiableMap((x: Double) => math.log(x), (x: Double) => 1.0 / x)
  )

  val h10: PushforwardMap[Double, Double, Double] = PushforwardMap(
    DataPipe((x: Double) => math.pow(10d, x)),
    DifferentiableMap(
      (x: Double) => math.log10(x),
      (x: Double) => 1.0 / (x * math.log(10d))
    )
  )

  val g1 = GaussianRV(0.0, 0.75)

  val g2 = GaussianRV(0.2, 0.75)

  val lg_p = h -> g1
  val lg_e = h -> g2

  val lu_reg = h10 -> UniformRV(-4d, -2.5d)

  val hyp_scaling = hyper_prior.map(
    p =>
      (
        p._1,
        Encoder(
          (x: Double) => (x - p._2.min) / (p._2.max - p._2.min),
          (u: Double) => u * (p._2.max - p._2.min) + p._2.min
        )
      )
  )

  val logit =
    Encoder((x: Double) => math.log(x / (1d - x)), (x: Double) => sigmoid(x))

  val hyp_mapping = Some(
    hyper_parameters
      .map(
        h => (h, hyp_scaling(h) > logit)
      )
      .toMap
  )

  val fitness_to_scalar =
    DataPipe[Seq[Tensor[Float]], Double](s => {
      val metrics = s.map(_.scalar.toDouble)
      metrics(2) / (metrics.head * metrics.head) - 2 * math
        .pow(metrics(1) / metrics.head, 2)
    })

  val loss_func_generator = (h: Map[String, Double]) => {

    val lossFunc = timelag.utils.get_pdt_loss[Double, Double, Double](
      causal_window._2,
      h("sigma_sq"),
      h("alpha")
    )

    val reg =
      if (reg_type == "L2")
        L2Regularization[Double](
          layer_scopes :+ output_scope,
          layer_parameter_names :+ "Outputs/Weights",
          layer_datatypes :+ "FLOAT64",
          layer_shapes :+ Shape(network_size.last, causal_window._2),
          math.exp(h("reg")),
          "L2Reg"
        )
      else
        L1Regularization[Double](
          layer_scopes :+ output_scope,
          layer_parameter_names :+ "Outputs/Weights",
          layer_datatypes :+ "FLOAT64",
          layer_shapes :+ Shape(network_size.last, causal_window._2),
          math.exp(h("reg")),
          "L1Reg"
        )

    lossFunc >>
      reg >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")
  }

  fte.exp_cdt_alt(
    architecture,
    hyper_parameters,
    persistent_hyper_parameters,
    params_enc,
    loss_func_generator,
    hyper_prior,
    hyp_mapping = hyp_mapping,
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
    pdt_iterations = pdt_iterations,
    latitude_limit = crop_latitude,
    fraction_pca = fraction_pca,
    deltaTFTE = history_fte,
    fteStep = fte_step,
    conv_flag = conv_flag,
    log_scale_fte = log_scale_fte,
    log_scale_omni = log_scale_omni,
    deltaT = causal_window,
    summary_top_dir = summary_dir,
    hyp_opt_iterations = hyp_opt_iterations,
    get_training_preds = get_training_preds,
    existing_exp = existing_exp,
    fitness_to_scalar = fitness_to_scalar,
    checkpointing_freq = checkpointing_freq
  )
}
