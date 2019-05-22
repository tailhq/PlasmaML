import _root_.org.joda.time._
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
import breeze.linalg.{DenseVector, DenseMatrix}
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
    timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  history_fte: Int = 10,
  fte_step: Int = 2,
  crop_latitude: Double = 40d,
  fraction_pca: Double = 0.8,
  log_scale_fte: Boolean = false,
  log_scale_omni: Boolean = false,
  conv_flag: Boolean = false,
  quantity: Int = OMNIData.Quantities.V_SW,
  use_persistence: Boolean = false,
  max_iterations: Int = 100000,
  max_iterations_tuning: Int = 20000,
  num_samples: Int = 4,
  hyper_optimizer: String = "gs",
  batch_size: Int = 32,
  optimization_algo: tf.train.Optimizer = tf.train.Adam(0.01f),
  summary_dir: Path = home / 'tmp,
  hyp_opt_iterations: Option[Int] = Some(5),
  reg_type: String = "L2",
  existing_exp: Option[Path] = None,
  checkpointing_freq: Int = 1
): helios.Experiment[Double, fte.ModelRunTuningSO[DenseVector[Double]], fte.data.FteOmniConfig] = {

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      -1,
      1,
      network_size,
      "FLOAT64"
    )

  val hyper_parameters = List(
    "reg"
  )

  val hyper_prior = Map(
    "reg" -> UniformRV(-5d, -3d)
  )

  //Prediction architecture
  val architecture =
    dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail)

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

  val output_scope = scope("Outputs")

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


  val loss_func_generator = (h: Map[String, Double]) => {

    val lossFunc = tf.learn.L2Loss[Double, Double]("Loss/L2Error")

    val reg =
      if (reg_type == "L2")
        L2Regularization[Double](
          layer_scopes,
          layer_parameter_names,
          layer_datatypes,
          layer_shapes,
          math.exp(h("reg")),
          "L2Reg"
        )
      else
        L1Regularization[Double](
          layer_scopes,
          layer_parameter_names,
          layer_datatypes,
          layer_shapes,
          math.exp(h("reg")),
          "L1Reg"
        )

    lossFunc >>
      tf.learn.Mean[Double]("Loss/Mean") >>
      reg >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")
  }

  val fitness_func = Seq(
    DataPipe2[Output[Double], Output[Double], Output[Float]](
      (p, t) => p.subtract(t).square.sum(axes = -1).mean().castTo[Float]
    ),
    DataPipe2[Output[Double], Output[Double], Output[Float]](
      (p, t) => p.subtract(t).abs.sum(axes = -1).mean().castTo[Float]
    )
  )

  val fitness_to_scalar =
    DataPipe[Seq[Tensor[Float]], Double](s => {
      val metrics = s.map(_.scalar.toDouble)
      metrics.sum/metrics.length
    })

  fte.exp_single_output(
    architecture,
    hyper_parameters,
    loss_func_generator,
    hyper_prior,
    fitness_func,
    hyp_mapping = hyp_mapping,
    year_range = start_year to end_year,
    test_year = test_year,
    sw_threshold = sw_threshold,
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
    deltaT = 4 * 24,
    summary_top_dir = summary_dir,
    hyp_opt_iterations = hyp_opt_iterations,
    fitness_to_scalar = fitness_to_scalar,
    checkpointing_freq = checkpointing_freq
  )
}


def baseline(
  experiment: Path,
  network_size: Seq[Int] = Seq(100, 60),
  activation_func: Int => Layer[Output[Double], Output[Double]] = (i: Int) =>
    timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  max_iterations: Int = 100000,
  max_iterations_tuning: Int = 20000,
  num_samples: Int = 4,
  hyper_optimizer: String = "gs",
  batch_size: Int = 32,
  optimization_algo: tf.train.Optimizer = tf.train.Adam(0.01f),
  hyp_opt_iterations: Option[Int] = Some(5),
  reg_type: String = "L2",
  existing_exp: Option[Path] = None,
  checkpointing_freq: Int = 1
): helios.Experiment[Double, fte.ModelRunTuningSO[DenseVector[Double]], fte.data.FteOmniConfig] = {

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      -1,
      1,
      network_size,
      "FLOAT64"
    )

  val hyper_parameters = List(
    "reg"
  )

  val hyper_prior = Map(
    "reg" -> UniformRV(-5d, -3d)
  )

  //Prediction architecture
  val architecture =
    dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail)

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

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


  val loss_func_generator = (h: Map[String, Double]) => {

    val lossFunc = tf.learn.L2Loss[Double, Double]("Loss/L2Error")

    val reg =
      if (reg_type == "L2")
        L2Regularization[Double](
          layer_scopes,
          layer_parameter_names,
          layer_datatypes,
          layer_shapes,
          math.exp(h("reg")),
          "L2Reg"
        )
      else
        L1Regularization[Double](
          layer_scopes,
          layer_parameter_names,
          layer_datatypes,
          layer_shapes,
          math.exp(h("reg")),
          "L1Reg"
        )

    lossFunc >>
      tf.learn.Mean[Double]("Loss/Mean") >>
      reg >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")
  }

  val fitness_func = Seq(
    DataPipe2[Output[Double], Output[Double], Output[Float]](
      (p, t) => p.subtract(t).square.sum(axes = -1).mean().castTo[Float]
    ),
    DataPipe2[Output[Double], Output[Double], Output[Float]](
      (p, t) => p.subtract(t).abs.sum(axes = -1).mean().castTo[Float]
    )
  )

  val fitness_to_scalar =
    DataPipe[Seq[Tensor[Float]], Double](s => {
      val metrics = s.map(_.scalar.toDouble)
      metrics.sum/metrics.length
    })

  fte.exp_single_output_baseline(
    experiment,
    architecture,
    hyper_parameters,
    loss_func_generator,
    hyper_prior,
    fitness_func,
    hyp_mapping = hyp_mapping,
    optimizer = optimization_algo,
    miniBatch = batch_size,
    iterations = max_iterations,
    num_samples = num_samples,
    hyper_optimizer = hyper_optimizer,
    iterations_tuning = max_iterations_tuning,
    hyp_opt_iterations = hyp_opt_iterations,
    fitness_to_scalar = fitness_to_scalar,
    checkpointing_freq = checkpointing_freq
  )
}

