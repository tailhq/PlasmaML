import _root_.ammonite.ops._
import _root_.org.joda.time._
import _root_.org.joda.time.format.{DateTimeFormat, DateTimeFormatter}
import _root_.breeze.linalg.{DenseVector, DenseMatrix}
import breeze.math._
import breeze.numerics._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe._
import _root_.io.github.mandar2812.dynaml.optimization._
import _root_.io.github.mandar2812.dynaml.evaluation._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.data._
import _root_.io.github.mandar2812.dynaml.tensorflow.utils.GaussianScalerTF
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import _root_.io.github.mandar2812.dynaml.{utils => dutils}
import _root_.io.github.mandar2812.dynaml.utils._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{
  L2Regularization,
  L1Regularization
}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.ammonite.ops._
import breeze.numerics.sigmoid
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Activation
import org.platanios.tensorflow.api.learn.layers.Layer

import OMNIData.Quantities._

def solar_wind_time_series(
  start: DateTime,
  end: DateTime,
  solar_wind_params: List[Int] = List(V_SW, V_Lat, V_Lon, B_X, B_Y, B_Z)
): ZipDataSet[DateTime, DenseVector[Double]] = {

  val omni_data_path = pwd / 'data

  val load_omni_file =
    fileToStream >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        OMNIData.dateColumns ++ solar_wind_params,
        OMNIData.columnFillValues
      ) >
      OMNILoader.processWithDateTime >
      IterableDataPipe(
        (p: (DateTime, Seq[Double])) => p._2.forall(x => !x.isNaN)
      )

  dtfdata
    .dataset(start.getYear to end.getYear)
    .map(
      DataPipe(
        (i: Int) => omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
      )
    )
    .flatMap(load_omni_file)
    .to_zip(
      identityPipe[DateTime] * DataPipe[Seq[Double], DenseVector[Double]](
        xs => DenseVector(xs.toArray)
      )
    )

}

case class OmniPDTConfig(
  solar_wind_params: List[Int],
  target_quantity: Int,
  data_limits: (Int, Int),
  test_year: Int,
  causal_window: (Int, Int),
  timelag_prediction: String = "mode",
  fraction_variance: Double = 1d,
  log_scale_targets: Boolean = false)
    extends helios.Config

def setup_exp(
  summary_top_dir: Path,
  solar_wind_params: List[Int] = List(V_SW, V_Lat, V_Lon, B_X, B_Y, B_Z),
  target_quantity: Int = Dst,
  start_year: Int = 2014,
  end_year: Int = 2016,
  test_year: Int = 2015,
  causal_window: (Int, Int) = (2, 12),
  fraction_pca: Double = 1.0
) = {

  val adj_fraction_pca = math.min(math.abs(fraction_pca), 1d)

  val experiment_config = OmniPDTConfig(
    solar_wind_params,
    target_quantity,
    (start_year, end_year),
    test_year,
    causal_window,
    fraction_variance = adj_fraction_pca
  )

  type PATTERN = (DateTime, (Tensor[Double], Tensor[Double]))
  val start = new DateTime(start_year, 1, 1, 0, 0, 0)
  val end   = new DateTime(end_year, 12, 31, 23, 59, 59)

  val omni = solar_wind_time_series(start, end, solar_wind_params)
  val omni_ground =
    fte.data
      .load_solar_wind_data_bdv(start, end)(
        (causal_window._1 - 1, causal_window._2 + 1),
        false,
        target_quantity
      )

  val (test_start, test_end) = (
    new DateTime(test_year, 1, 1, 0, 0),
    new DateTime(test_year, 12, 31, 23, 59)
  )

  val tt_partition = DataPipe(
    (p: (DateTime, (DenseVector[Double], DenseVector[Double]))) =>
      if (p._1.isAfter(test_start) && p._1.isBefore(test_end))
        false
      else
        true
  )

  val difference_mat = DenseMatrix.tabulate(
    causal_window._2, causal_window._2 + 1)(
    (i, j) => if (i == j) -1d else if (i == j - 1) 1d else 0d)

  val difference_op = DataPipe((v: DenseVector[Double]) => difference_mat * v)

  val omni_pdt = omni.join(omni_ground).map(
    identityPipe[DateTime]* (identityPipe[DenseVector[Double]] * difference_op)
    ).partition(tt_partition)

  val sum_dir_prefix = "omni_pdt_"

  val dt = DateTime.now()

  val summary_dir_index = sum_dir_prefix + dt.toString("YYYY-MM-dd-HH-mm")

  val tf_summary_dir = summary_top_dir / summary_dir_index

  println("Serializing data sets")
  fte.data.write_data_set(
    dt.toString("YYYY-MM-dd-HH-mm"),
    omni_pdt,
    DataPipe[DenseVector[Double], Seq[Double]](_.toArray.toSeq),
    DataPipe[DenseVector[Double], Seq[Double]](_.toArray.toSeq),
    tf_summary_dir
  )
  (experiment_config, tf_summary_dir)
}

@main
def apply(
  solar_wind_params: List[Int] = List(V_SW, V_Lat, V_Lon, B_X, B_Y, B_Z),
  target_quantity: Int = Dst,
  start_year: Int = 2014,
  end_year: Int = 2016,
  test_year: Int = 2015,
  causal_window: (Int, Int) = (2, 12),
  fraction_pca: Double = 1.0,
  network_size: Seq[Int] = Seq(20, 20),
  activation_func: Int => Layer[Output[Double], Output[Double]] = (i: Int) =>
    timelag.utils.getReLUAct3[Double](1, 1, i, 0f),
  iterations: Int = 100000,
  iterations_tuning: Int = 20000,
  pdt_iterations_tuning: Int = 4,
  pdt_iterations_test: Int = 14,
  num_samples: Int = 4,
  hyper_optimizer: String = "gs",
  batch_size: Int = 32,
  optimizer: tf.train.Optimizer = tf.train.Adam(0.01f),
  summary_top_dir: Path = home / 'tmp,
  hyp_opt_iterations: Option[Int] = Some(5),
  get_training_preds: Boolean = false,
  reg_type: String = "L2",
  existing_exp: Option[Path] = None,
  checkpointing_freq: Int = 1
): helios.Experiment[Double, fte.ModelRunTuning[DenseVector[Double]], OmniPDTConfig] = {

  val causal_window_size = causal_window._2

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      -1,
      network_size.last,
      network_size.take(network_size.length - 1),
      "FLOAT64"
    )

  val output_mapping = {

    val outputs_segment =
      tf.learn.Linear[Double]("Outputs", causal_window_size)

    val timelag_segment =
      tf.learn.Linear[Double]("TimeLags", causal_window_size) >>
        tf.learn.Softmax[Double]("Probability/Softmax")

    dtflearn.bifurcation_layer("PDTNet", outputs_segment, timelag_segment)
  }

  val hyper_params = List(
    "sigma_sq",
    "alpha",
    "reg"
  )

  val persistent_hyp_params = List("reg")

  val hyper_prior = Map(
    "reg"      -> UniformRV(-5d, -3d),
    "alpha"    -> UniformRV(0.75d, 2d),
    "sigma_sq" -> UniformRV(1e-5, 5d)
  )

  val params_to_mutable_params = Encoder(
    identityPipe[Map[String, Double]],
    identityPipe[Map[String, Double]]
  )

  val architecture = dtflearn.feedforward_stack[Double](activation_func)(
    net_layer_sizes.tail
  ) >>
    activation_func(net_layer_sizes.tail.length) >>
    output_mapping

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
    hyper_params
      .map(
        h => (h, hyp_scaling(h) > logit)
      )
      .toMap
  )

  val fitness_to_scalar =
    DataPipe[Seq[Tensor[Float]], Double](s => {
      val metrics = s.map(_.scalar.toDouble)
      metrics(1) / metrics.head
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

  val (experiment_config, tf_summary_dir) = setup_exp(
    summary_top_dir,
    solar_wind_params,
    target_quantity,
    start_year,
    end_year,
    test_year,
    causal_window,
    fraction_pca
  )

  val results = fte.run_exp(
    tf_summary_dir,
    architecture,
    hyper_params,
    persistent_hyp_params,
    params_to_mutable_params,
    loss_func_generator,
    hyper_prior,
    hyp_mapping,
    iterations,
    iterations_tuning,
    pdt_iterations_tuning,
    pdt_iterations_test,
    num_samples,
    hyper_optimizer,
    batch_size,
    optimizer,
    hyp_opt_iterations,
    get_training_preds,
    existing_exp,
    fitness_to_scalar,
    checkpointing_freq,
    experiment_config.log_scale_targets
  )

  helios.Experiment(
    experiment_config,
    results
  )

}
