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

import _root_.org.json4s._
import _root_.org.json4s.JsonDSL._
import _root_.org.json4s.jackson.Serialization.{
  read => read_json,
  write => write_json
}
import org.json4s.jackson.JsonMethods._

DateTimeZone.setDefault(DateTimeZone.UTC)

implicit val formats = DefaultFormats + FieldSerializer[Map[String, Any]]()

implicit val dateOrdering: Ordering[DateTime] = new Ordering[DateTime] {
  override def compare(x: DateTime, y: DateTime): Int =
    if (x.isBefore(y)) -1 else 1
}

case class OmniMexConfig(
  data_limits: (Int, Int),
  test_year: Int,
  causal_window: (Int, Int),
  timelag_prediction: String = "mode",
  fraction_variance: Double = 1d)
    extends helios.Config

type ModelRunTuning = helios.TunedModelRun2[
  DenseVector[Double],
  DenseVector[Double],
  Double,
  Output[Double],
  (Output[Double], Output[Double]),
  Double,
  Tensor[Double],
  FLOAT64,
  Shape,
  (Tensor[Double], Tensor[Double]),
  (FLOAT64, FLOAT64),
  (Shape, Shape)
]

//Linear Segment is a case class for performing
//linear interpolation based data imputation
case class LinearSegment(
  time_limits: (DateTime, DateTime),
  values: (Double, Double)) {

  require(
    time_limits._1.isBefore(time_limits._2),
    "In a Linear Interpolation Segment, the date time nodes must be ordered correctly"
  )

  val (start_ts, end_ts) =
    (time_limits._1.getMillis(), time_limits._2.getMillis())

  val deltaT = end_ts - start_ts

  val deltaV = values._2 - values._1

  def apply(t: DateTime): Double =
    if (t.isBefore(time_limits._1) || t.isAfter(time_limits._2)) 0d
    else values._1 + (t.getMillis() - start_ts) * deltaV / deltaT

}

def solar_wind_time_series(
  start: DateTime,
  end: DateTime
): ZipDataSet[DateTime, DenseVector[Double]] = {

  val omni_data_path = pwd / 'data

  val load_omni_file =
    fileToStream >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        OMNIData.dateColumns ++ List(V_SW, V_Lat, V_Lon, B_X, B_Y, B_Z, F10_7),
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

def mex_solar_wind_time_series(
  files: Seq[Path],
  start: DateTime,
  end: DateTime,
  forward_causal_window: (Int, Int)
): ZipDataSet[DateTime, DenseVector[Double]] = {

  val dt_format: DateTimeFormatter =
    DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss.SSS")

  val read_file = DataPipe[Path, String](_.toString) > fileToStream

  val non_header_lines = IterableDataPipe((line: String) => line.head != '#')

  val dt_roundoff: DataPipe[DateTime, DateTime] = DataPipe(
    (d: DateTime) =>
      new DateTime(
        d.getYear,
        d.getMonthOfYear,
        d.getDayOfMonth,
        d.getHourOfDay,
        0,
        0
      )
  )

  val process_file =
    read_file >
      non_header_lines >
      replaceWhiteSpaces >
      IterableDataPipe((l: String) => {
        val splits = l.split(",")

        (
          DateTime.parse(splits.head, dt_format),
          try {
            splits.last.toDouble
          } catch {
            case _: Exception => Double.NaN
          }
        )
      }) >
      IterableDataPipe(dt_roundoff * identityPipe[Double]) >
      IterableDataPipe(
        tup2_1[DateTime, Double] > DataPipe[DateTime, Boolean](
          d => d.isAfter(start) && d.isBefore(end)
        )
      )

  val group_by_hour = DataPipe(
    (ds: Iterable[(DateTime, Double)]) =>
      ds.groupBy(_._1).toIterable.map(p => (p._1, p._2.map(_._2).toSeq))
  )

  val sort_by_dt = DataPipe(
    (ds: Iterable[(DateTime, Seq[Double])]) => ds.toSeq.sortBy(_._1).toIterable
  )

  val hourly_median = DataPipe[Seq[Double], Double](
    xs =>
      if (xs.isEmpty || xs.filterNot(_.isNaN).isEmpty) Double.NaN
      else dutils.median(xs.filterNot(_.isNaN).toStream.sorted)
  )

  val interpolate = DataPipe(
    (xs: Iterable[(DateTime, Double)]) =>
      xs.sliding(2)
        .toIterable
        .flatMap(interval => {
          val duration =
            new Duration(interval.head._1, interval.last._1).getStandardHours

          if (duration > 1) {
            val lin_segment = LinearSegment(
              (interval.head._1, interval.last._1),
              (interval.head._2, interval.last._2)
            )

            (0 until duration.toInt).toIterable.map(l => {
              val t = interval.head._1.plusHours(l)
              (t, lin_segment(t))
            })

          } else Iterable(interval.head)
        })
  )

  val construct_causal_windows = DataPipe(
    (xs: Iterable[(DateTime, Double)]) =>
      xs.sliding(forward_causal_window._1 + forward_causal_window._2)
        .toIterable
        .map(window => {

          (
            window.head._1,
            window.takeRight(forward_causal_window._2).map(_._2).toSeq
          )
        })
  )

  dtfdata
    .dataset(files)
    .flatMap(process_file > group_by_hour > sort_by_dt)
    .map(identityPipe[DateTime] * hourly_median)
    .filterNot(DataPipe[(DateTime, Double), Boolean](_._2.isNaN))
    .transform(interpolate > construct_causal_windows)
    .to_zip(
      identityPipe[DateTime] * DataPipe[Seq[Double], DenseVector[Double]](
        xs => DenseVector(xs.toArray)
      )
    )

}

def dump_omni_mex_data(
  start_year: Int,
  end_year: Int,
  causal_window: (Int, Int),
  file: Path
): Unit = {

  val start = new DateTime(start_year, 1, 1, 0, 0, 0)
  val end   = new DateTime(end_year, 12, 31, 23, 59, 59)

  val omni = solar_wind_time_series(start, end)
  val mex_sw = mex_solar_wind_time_series(
    Seq(home / 'Downloads / "mex_v_sw.txt"),
    start,
    end,
    causal_window
  )

  val omni_mex = omni.join(mex_sw)

  if (exists ! file) rm ! file

  println("Writing data sets")

  val pattern_to_map =
    DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), JValue](
      p =>
        (
          ("timestamp" -> p._1.toString("yyyy-MM-dd'T'HH:mm:ss'Z'")) ~
            ("targets" -> p._2._2.toArray.toList) ~
            ("inputs"  -> p._2._1.toArray.toList)
        )
    )

  val map_to_json = DataPipe[JValue, String](p => write_json(p))

  val process_pattern = pattern_to_map > map_to_json

  val write_pattern: String => Unit =
    line =>
      write.append(
        file,
        s"${line}\n"
      )

  omni_mex
    .map(process_pattern)
    .data
    .foreach(write_pattern)

}

def read_omni_mex_data_set(
  data_file: Path
): DataSet[(DateTime, (DenseVector[Double], DenseVector[Double]))] = {

  require(
    exists ! data_file,
    "Both training and test files must exist."
  )

  val dt_format = DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss'Z'")

  val read_file = DataPipe((p: Path) => read.lines ! p)

  val filter_non_empty_lines = IterableDataPipe((l: String) => !l.isEmpty)

  val read_json_record = IterableDataPipe((s: String) => parse(s))

  val load_record = IterableDataPipe((record: JValue) => {
    val dt = dt_format.parseDateTime(
      record
        .findField(p => p._1 == "timestamp")
        .get
        ._2
        .values
        .asInstanceOf[String]
    )
    val features = DenseVector(
      record
        .findField(p => p._1 == "inputs")
        .get
        ._2
        .values
        .asInstanceOf[List[Double]]
        .toArray
    )

    val targets_seq = DenseVector(
      record
        .findField(p => p._1 == "targets")
        .get
        ._2
        .values
        .asInstanceOf[List[Double]]
        .toArray
    )

    (dt, (features, targets_seq))
  })

  val pipeline = read_file > filter_non_empty_lines > read_json_record > load_record

  dtfdata
    .dataset(Seq(data_file))
    .flatMap(pipeline)
    .to_zip(
      identityPipe[(DateTime, (DenseVector[Double], DenseVector[Double]))]
    )
}

type SC_DATA = (
  TFDataSet[(DateTime, (DenseVector[Double], DenseVector[Double]))],
  (Scaler[DenseVector[Double]], GaussianScaler)
)

val scale_bdv_data =
  DataPipe(
    (dataset: TFDataSet[
      (DateTime, (DenseVector[Double], DenseVector[Double]))
    ]) => {

      type P = (DateTime, (DenseVector[Double], DenseVector[Double]))

      val features = dataset.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_1[
            DenseVector[Double],
            DenseVector[Double]
          ]
        )
        .data

      val targets = dataset.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_2[
            DenseVector[Double],
            DenseVector[Double]
          ]
        )
        .data

      val gaussian_scaling = DataPipe[
        Iterable[DenseVector[Double]],
        GaussianScaler
      ](ds => {
        val (mean, variance) = dutils.getStats(ds)
        GaussianScaler(mean, sqrt(variance))
      })

      val targets_scaler = gaussian_scaling(targets)

      val features_scaler = gaussian_scaling(features)

      val scale_training_data = identityPipe[DateTime] * (features_scaler * targets_scaler)
      val scale_test_data = identityPipe[DateTime] * (features_scaler * identityPipe[
        DenseVector[Double]
      ])

      (
        dataset.copy(
          training_dataset = dataset.training_dataset.map(scale_training_data),
          test_dataset = dataset.test_dataset.map(scale_test_data)
        ),
        (features_scaler, targets_scaler)
      )

    }
  )

def process_predictions(
  scaled_data: DataSet[(DateTime, (DenseVector[Double], DenseVector[Double]))],
  predictions: (Tensor[Double], Tensor[Double]),
  scalers: (Scaler[DenseVector[Double]], GaussianScaler),
  causal_window: Int,
  mo_flag: Boolean,
  prob_timelags: Boolean,
  log_scale_omni: Boolean,
  scale_actual_targets: Boolean = true
) = {

  val nTest = scaled_data.size

  val scaler_tf = GaussianScalerTF(
    dtf.tensor_f64(causal_window)(scalers._2.mean.toArray: _*),
    dtf.tensor_f64(causal_window)(scalers._2.sigma.toArray: _*)
  )

  val index_times = Tensor(
    (0 until causal_window).map(_.toDouble)
  ).reshape(
    Shape(causal_window)
  )

  val pred_time_lags_test: Tensor[Double] = if (prob_timelags) {
    val unsc_probs = predictions._2

    unsc_probs.topK(1)._2.reshape(Shape(nTest)).castTo[Double]

  } else predictions._2

  val unscaled_preds_test = scaler_tf.i(predictions._1)

  val pred_targets_test: Tensor[Double] = if (mo_flag) {

    val repeated_times =
      tfi.stack(Seq.fill(causal_window)(pred_time_lags_test.floor), axis = -1)

    val conv_kernel =
      repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

    unscaled_preds_test
      .multiply(conv_kernel)
      .sum(axes = 1)
      .divide(conv_kernel.sum(axes = 1))
  } else {
    scaler_tf(0).i(predictions._1)
  }

  val test_labels = scaled_data.data
    .map(_._2._2)
    .map(
      t =>
        if (scale_actual_targets) scalers._2.i(t).toArray.toSeq
        else t.toArray.toSeq
    )
    .toSeq

  val actual_targets = test_labels.zipWithIndex.map(zi => {
    val (z, index) = zi
    val time_lag =
      pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

    z(time_lag)
  })

  val (final_predictions, final_targets) =
    if (log_scale_omni) (pred_targets_test.exp, actual_targets.map(math.exp))
    else (pred_targets_test, actual_targets)

  (final_predictions, final_targets, unscaled_preds_test, pred_time_lags_test)
}

@main
def apply(
  data_file: Path,
  start_year: Int = 2014,
  end_year: Int = 2016,
  test_year: Int = 2015,
  causal_window: (Int, Int) = (8, 16),
  fraction_pca: Double = 1.0,
  network_size: Seq[Int] = Seq(100, 60),
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
): helios.Experiment[Double, ModelRunTuning, OmniMexConfig] = {

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
      //tf.learn.BatchNormalization[Double]("BatchNorm", fused = false) >>
      tf.learn.Linear[Double]("Outputs", causal_window_size)

    val timelag_segment =
      tf.learn.Linear[Double]("TimeLags", causal_window_size) >>
        tf.learn.Softmax[Double]("Probability/Softmax")

    dtflearn.bifurcation_layer("PDTNet", outputs_segment, timelag_segment)
  }

  val hyper_parameters = List(
    "sigma_sq",
    "alpha",
    "reg"
  )

  val persistent_hyper_parameters = List("reg")

  val hyper_prior = Map(
    "reg"      -> UniformRV(-5d, -2.5d),
    "alpha"    -> UniformRV(0.75d, 2d),
    "sigma_sq" -> UniformRV(1e-5, 5d)
  )

  val params_enc = Encoder(
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
    hyper_parameters
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

  type PATTERN = (DateTime, (DenseVector[Double], DenseVector[Double]))
  val start = new DateTime(start_year, 1, 1, 0, 0, 0)
  val end   = new DateTime(end_year, 12, 31, 23, 59, 59)

  /*val omni = solar_wind_time_series(start, end)
  val mex_sw = mex_solar_wind_time_series(
    Seq(home / 'Downloads / "mex_v_sw.txt"),
    start,
    end,
    causal_window
  ) */

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

  val omni_mex = read_omni_mex_data_set(data_file)
    .filter(
      tup2_1[DateTime, (DenseVector[Double], DenseVector[Double])] > DataPipe[
        DateTime,
        Boolean
      ](
        d => d.isAfter(start) && d.isBefore(end)
      )
    )
    .partition(tt_partition) //omni.join(mex_sw).partition(tt_partition)

  val sum_dir_prefix = "omni_mex"

  val dt = DateTime.now()

  val mo_flag: Boolean       = true
  val prob_timelags: Boolean = true

  val summary_dir_index = {
    if (mo_flag) sum_dir_prefix + "_mo_tl_" + dt.toString("YYYY-MM-dd-HH-mm")
    else sum_dir_prefix + "_tl_" + dt.toString("YYYY-MM-dd-HH-mm")
  }

  val tf_summary_dir = summary_top_dir / summary_dir_index

  val adj_fraction_pca = math.min(math.abs(fraction_pca), 1d)

  val data_size = omni_mex.training_dataset.size

  val scaling_op = scale_bdv_data

  println("Scaling data attributes")
  val (scaled_data, scalers): SC_DATA = scaling_op.run(omni_mex)

  val split_data = scaled_data.training_dataset.partition(
    DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Boolean](
      _ => scala.util.Random.nextDouble() <= 0.7
    )
  )

  val input_shape  = Shape(scaled_data.training_dataset.data.head._2._1.length)
  val output_shape = Shape(scaled_data.training_dataset.data.head._2._2.length)

  val load_pattern_in_tensor =
    tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] >
      (
        DataPipe(
          (dv: DenseVector[Double]) =>
            dtf.tensor_f64(input_shape(0))(dv.toArray.toSeq: _*)
        ) *
          DataPipe(
            (dv: DenseVector[Double]) =>
              dtf.tensor_f64(output_shape(0))(dv.toArray.toSeq: _*)
          )
      )

  val unzip =
    DataPipe[Iterable[(Tensor[Double], Tensor[Double])], (Iterable[Tensor[Double]], Iterable[Tensor[Double]])](
      _.unzip
    )

  val concatPreds = unzip > (dtfpipe.EagerConcatenate[Double](axis = 0) * dtfpipe
    .EagerConcatenate[Double](axis = 0))

  val tf_handle_ops_tuning = dtflearn.model.tf_data_handle_ops[
    (DateTime, (DenseVector[Double], DenseVector[Double])),
    Tensor[Double],
    Tensor[Double],
    (Tensor[Double], Tensor[Double]),
    Output[Double],
    Output[Double]
  ](
    patternToTensor = Some(load_pattern_in_tensor),
    concatOpI = Some(dtfpipe.EagerStack[Double](axis = 0)),
    concatOpT = Some(dtfpipe.EagerStack[Double](axis = 0))
  )

  val tf_handle_ops_test = dtflearn.model.tf_data_handle_ops[
    (DateTime, (DenseVector[Double], DenseVector[Double])),
    Tensor[Double],
    Tensor[Double],
    (Tensor[Double], Tensor[Double]),
    Output[Double],
    Output[Double]
  ](
    patternToTensor = Some(load_pattern_in_tensor),
    concatOpI = Some(dtfpipe.EagerStack[Double](axis = 0)),
    concatOpT = Some(dtfpipe.EagerStack[Double](axis = 0)),
    concatOpO = Some(concatPreds)
  )

  val tf_data_ops: dtflearn.model.Ops[Output[Double], Output[Double]] =
    dtflearn.model.data_ops(
      shuffleBuffer = 10,
      batchSize = batch_size,
      prefetchSize = 10
    )

  val config_to_dir = DataPipe[Map[String, Double], String](
    _.map(kv => s"${kv._1}#${kv._2}").mkString("_")
  )

  val (adjusted_iterations, adjusted_iterations_tuning) = (
    iterations / (pdt_iterations_test + 1),
    iterations_tuning / (pdt_iterations_tuning + 1)
  )

  val train_config_tuning = dtflearn.tunable_tf_model.ModelConfigFunction(
    DataPipe[Map[String, Double], Path](
      h =>
        dtflearn.tunable_tf_model.ModelFunction.get_summary_dir(
          tf_summary_dir,
          h,
          Some(config_to_dir)
        )
    ),
    DataPipe[Map[String, Double], dtflearn.model.Ops[Output[Double], Output[
      Double
    ]]](_ => tf_data_ops),
    DataPipe((_: Map[String, Double]) => optimizer),
    DataPipe(
      (_: Map[String, Double]) =>
        dtflearn.rel_loss_change_stop(0.005, adjusted_iterations_tuning)
    ),
    DataPipe(
      (h: Map[String, Double]) =>
        Some(
          timelag.utils.get_train_hooks(
            tf_summary_dir / config_to_dir(h),
            adjusted_iterations_tuning,
            false,
            data_size,
            batch_size,
            checkpointing_freq * 2,
            checkpointing_freq
          )
        )
    )
  )

  val train_config_test =
    dtflearn.model.trainConfig(
      summaryDir = tf_summary_dir,
      data_processing = tf_data_ops,
      optimizer = optimizer,
      stopCriteria = dtflearn.rel_loss_change_stop(0.005, adjusted_iterations),
      trainHooks = Some(
        timelag.utils.get_train_hooks(
          tf_summary_dir,
          adjusted_iterations,
          false,
          data_size,
          batch_size,
          checkpointing_freq * 2,
          checkpointing_freq
        )
      )
    )

  val model_function =
    dtflearn.tunable_tf_model.ModelFunction.from_loss_generator[
      Output[Double],
      Output[Double],
      (Output[Double], Output[Double]),
      Double,
      Tensor[Double],
      DataType[Double],
      Shape,
      Tensor[Double],
      DataType[Double],
      Shape,
      (Tensor[Double], Tensor[Double]),
      (DataType[Double], DataType[Double]),
      (Shape, Shape)
    ](
      loss_func_generator,
      architecture,
      (FLOAT64, input_shape),
      (FLOAT64, Shape(causal_window_size))
    )

  val pdtModel = helios.learn.pdt_model(
    causal_window_size,
    model_function,
    train_config_tuning,
    hyper_parameters,
    persistent_hyper_parameters,
    params_enc,
    split_data.training_dataset,
    tf_handle_ops_tuning,
    fitness_to_scalar = fitness_to_scalar,
    validation_data = Some(split_data.test_dataset)
  )

  val gs = hyper_optimizer match {
    case "csa" =>
      new CoupledSimulatedAnnealing[pdtModel.type](
        pdtModel,
        hyp_mapping
      ).setMaxIterations(
        hyp_opt_iterations.getOrElse(5)
      )

    case "gs" => new GridSearch[pdtModel.type](pdtModel)

    case "cma" =>
      new CMAES[pdtModel.type](
        pdtModel,
        hyper_parameters,
        learning_rate = 0.8,
        hyp_mapping
      ).setMaxIterations(hyp_opt_iterations.getOrElse(5))

    case _ => new GridSearch[pdtModel.type](pdtModel)
  }

  gs.setPrior(hyper_prior)

  gs.setNumSamples(num_samples)

  println(
    "--------------------------------------------------------------------"
  )
  println("Initiating model tuning")
  println(
    "--------------------------------------------------------------------"
  )

  val (_, config) = gs.optimize(
    hyper_prior.mapValues(_.draw),
    Map(
      "loops"       -> pdt_iterations_tuning.toString,
      "evalTrigger" -> (adjusted_iterations_tuning / checkpointing_freq).toString
    )
  )

  println(
    "--------------------------------------------------------------------"
  )
  println("\nModel tuning complete")
  println("Chosen configuration:")
  pprint.pprintln(config)
  println(
    "--------------------------------------------------------------------"
  )

  println("Training final model based on chosen configuration")

  println("Chosen configuration:")
  pprint.pprintln(config)
  println(
    "--------------------------------------------------------------------"
  )

  println("Training final model based on chosen configuration")

  val (best_model, best_config) = pdtModel.build(
    pdt_iterations_test,
    config,
    Some(train_config_test),
    Some(adjusted_iterations / checkpointing_freq)
  )

  val chosen_config = config.filterKeys(persistent_hyper_parameters.contains) ++ best_config

  write.over(
    tf_summary_dir / "state.csv",
    chosen_config.keys.mkString(start = "", sep = ",", end = "\n") +
      chosen_config.values.mkString(start = "", sep = ",", end = "")
  )

  val extract_tensors = load_pattern_in_tensor

  val extract_features = tup2_1[Tensor[Double], Tensor[Double]]

  val model_predictions_test = best_model.infer_batch(
    scaled_data.test_dataset.map(extract_tensors > extract_features),
    train_config_test.data_processing,
    tf_handle_ops_test
  )

  val predictions = model_predictions_test match {
    case Left(tensor)      => tensor
    case Right(collection) => timelag.utils.collect_predictions(collection)
  }

  val (
    final_predictions,
    final_targets,
    unscaled_preds_test,
    pred_time_lags_test
  ) = process_predictions(
    scaled_data.test_dataset,
    predictions,
    scalers,
    causal_window_size,
    mo_flag,
    prob_timelags,
    false,
    false
  )

  val reg_metrics = new RegressionMetricsTF(final_predictions, final_targets)

  val (reg_metrics_train, preds_train) = if (get_training_preds) {

    val model_predictions_train = best_model.infer_batch(
      scaled_data.training_dataset.map(extract_tensors > extract_features),
      train_config_test.data_processing,
      tf_handle_ops_test
    )

    val predictions_train = model_predictions_train match {
      case Left(tensor)      => tensor
      case Right(collection) => timelag.utils.collect_predictions(collection)
    }

    val (
      final_predictions_train,
      final_targets_train,
      unscaled_preds_train,
      pred_time_lags_train
    ) = process_predictions(
      scaled_data.training_dataset,
      predictions_train,
      scalers,
      causal_window_size,
      mo_flag,
      prob_timelags,
      false,
      true
    )

    val reg_metrics_train =
      new RegressionMetricsTF(final_predictions_train, final_targets_train)

    println("Writing model predictions: Training Data")
    helios.write_predictions[Double](
      (unscaled_preds_train, predictions_train._2),
      tf_summary_dir,
      "train_" + dt.toString("YYYY-MM-dd-HH-mm")
    )

    helios.write_processed_predictions(
      dtfutils.toDoubleSeq(final_predictions_train).toSeq,
      final_targets_train,
      dtfutils.toDoubleSeq(pred_time_lags_train).toSeq,
      tf_summary_dir / ("scatter_train-" + dt
        .toString("YYYY-MM-dd-HH-mm") + ".csv")
    )

    println("Writing performance results: Training Data")
    helios.write_performance(
      "train_" + dt.toString("YYYY-MM-dd-HH-mm"),
      reg_metrics_train,
      tf_summary_dir
    )

    (
      Some(reg_metrics_train),
      Some(final_predictions_train, pred_time_lags_train)
    )
  } else {
    (None, None)
  }

  val results = helios.TunedModelRun2(
    (scaled_data, scalers),
    best_model,
    reg_metrics_train,
    Some(reg_metrics),
    tf_summary_dir,
    preds_train,
    Some((final_predictions, pred_time_lags_test))
  )

  println("Writing model predictions: Test Data")
  helios.write_predictions[Double](
    (unscaled_preds_test, predictions._2),
    tf_summary_dir,
    "test_" + dt.toString("YYYY-MM-dd-HH-mm")
  )

  helios.write_processed_predictions(
    dtfutils.toDoubleSeq(final_predictions).toSeq,
    final_targets,
    dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
    tf_summary_dir / ("scatter_test-" + dt
      .toString("YYYY-MM-dd-HH-mm") + ".csv")
  )

  println("Writing performance results: Test Data")
  helios.write_performance(
    "test_" + dt.toString("YYYY-MM-dd-HH-mm"),
    reg_metrics,
    tf_summary_dir
  )

  helios.Experiment(
    OmniMexConfig(
      (start_year, end_year),
      test_year,
      causal_window,
      fraction_variance = fraction_pca
    ),
    results
  )

}
