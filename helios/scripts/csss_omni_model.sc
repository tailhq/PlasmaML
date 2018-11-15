import ammonite.ops._
import org.joda.time._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.evaluation._
import io.github.mandar2812.dynaml.tensorflow.data._
import io.github.mandar2812.dynaml.tensorflow.utils._
import io.github.mandar2812.dynaml.tensorflow.{dtf, dtfdata, dtflearn, dtfutils}
import io.github.mandar2812.dynaml.tensorflow.implicits._
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import _root_.io.github.mandar2812.PlasmaML.utils.L2Regularization
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.platanios.tensorflow.api.learn.layers.Activation
import $file.timelagutils

//Set time zone to UTC
DateTimeZone.setDefault(DateTimeZone.UTC)

//Load the Carrington Rotation Table
val carrington_rotation_table = pwd/'data/"CR_Table.rdb"

val process_carrington_file =
  DataPipe((p: Path) => (read.lines! p).toStream) > dropHead > dropHead > trimLines > replaceWhiteSpaces > splitLine

case class CarringtonRotation(
  start: DateTime,
  end: DateTime) {

  def contains(dt: DateTime): Boolean = dt.isAfter(start) && dt.isBefore(end)
}

val read_time_stamps = DataPipe((s: Array[String]) => {

  val datetime_pattern = "YYYY.MM.dd_HH:mm:ss"
  val dt = format.DateTimeFormat.forPattern(datetime_pattern)

  val limits = (DateTime.parse(s(1), dt), DateTime.parse(s(3), dt))

  (s.head.toInt, CarringtonRotation(limits._1, limits._2))
})

val carrington_rotations = dtfdata.dataset(process_carrington_file(carrington_rotation_table)).to_zip(read_time_stamps)



val fte_file = MetaPipe(
  (data_path: Path) => (carrington_rotation: Int) => data_path/s"HMIfootpoint_ch_csss${carrington_rotation}HR.dat"
)


case class FTEPattern(
  longitude: Double,
  latitude: Double,
  fte: Option[Double])

val process_fte_file = {
  fte_file >> (
    DataPipe((p: Path) => (read.lines! p).toStream) >
      Seq.fill(4)(dropHead).reduceLeft(_ > _) >
      trimLines >
      replaceWhiteSpaces >
      splitLine >
      IterableDataPipe((s: Array[String]) => s.length == 5) >
      IterableDataPipe((s: Array[String]) => {
        val (lon, lat) = (s.head.toDouble, s(1).toDouble)
        val fte: Option[Double] = try {
          Some(s(2).toDouble)
        } catch {
          case _: Exception => None
        }

        FTEPattern(lon, lat, fte)

      })
    )

}


def get_fte_for_rotation(data_path: Path)(cr: Int) = try {
  Iterable((cr, process_fte_file(data_path)(cr)))
} catch {
  case _: java.nio.file.NoSuchFileException => Iterable()
}


val process_rotation = DataPipe((rotation_data: (Int, (CarringtonRotation, Iterable[FTEPattern]))) => {

  val (_, (rotation, fte)) = rotation_data

  val duration = new Duration(rotation.start, rotation.end)

  val time_jump = duration.getMillis/360.0

  fte.map(p => (rotation.end.toInstant.minus((time_jump*p.longitude).toLong).toDateTime, p))

})

val image_dt_roundoff: DataPipe[DateTime, DateTime] = DataPipe((d: DateTime) =>
  new DateTime(
    d.getYear, d.getMonthOfYear,
    d.getDayOfMonth, d.getHourOfDay,
    0, 0)
)

implicit val dateOrdering: Ordering[DateTime] = new Ordering[DateTime] {
  override def compare(x: DateTime, y: DateTime): Int = if(x.isBefore(y)) -1 else 1
}


def load_fte_data(
  data_path: Path,
  carrington_rotation_table: ZipDataSet[Int, CarringtonRotation],
  log_flag: Boolean)(
  start: DateTime, end: DateTime): ZipDataSet[DateTime, Tensor] = {

  val start_rotation = carrington_rotation_table.filter(_._2.contains(start)).data.head._1

  val end_rotation = carrington_rotation_table.filter(_._2.contains(end)).data.head._1

  val fte = dtfdata.dataset(start_rotation to end_rotation)
    .flatMap(get_fte_for_rotation(data_path) _)
    .to_zip(identityPipe)

  val fte_data = carrington_rotation_table.join(fte)

  val log_transformation =
    (x: Double) => if(log_flag) {
      if(math.abs(x) < 1d) 0d
      else math.log10(math.abs(x))
    } else x

  val processed_fte_data = {
    fte_data.flatMap(process_rotation)
      .transform(
        (data: Iterable[(DateTime, FTEPattern)]) =>
          data.groupBy(_._1).map(p => (p._1, p._2.map(_._2).toSeq.sortBy(_.latitude))))
      .filter(_._2.length == 180)
      .map(image_dt_roundoff * identityPipe[Seq[FTEPattern]])
      .transform((s: Iterable[(DateTime, Seq[FTEPattern])]) => s.toSeq.sortBy(_._1))
      .to_zip(
        identityPipe[DateTime] *
          DataPipe((s: Seq[FTEPattern]) =>
            Tensor(s.map(_.fte.get).map(log_transformation)).reshape(Shape(s.length))
          )
      )
  }

  println("Interpolating FTE values to fill hourly cadence requirement")
  val interpolated_fte = dtfdata.dataset(
    processed_fte_data.data.sliding(2)
      .filter(p => new Duration(p.head._1, p.last._1).getStandardHours > 1)
      .flatMap(i => {
        val duration = new Duration(i.head._1, i.last._1).getStandardHours
        val delta_fte = (i.last._2 - i.head._2)/duration.toDouble

        (1 until duration.toInt).map(l => (i.head._1.plusHours(l), i.head._2 + delta_fte*l))
      }).toIterable
  )


  processed_fte_data
    .concatenate(interpolated_fte)
    .transform((data: Iterable[(DateTime, Tensor)]) => data.toSeq.sortBy(_._1))
    .to_zip(identityPipe[(DateTime, Tensor)])

}


def load_solar_wind_data(start: DateTime, end: DateTime)(deltaT: (Int, Int)) = {
  val omni_processing =
    OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(OMNIData.Quantities.V_SW) >
      IterableDataPipe[(DateTime, Seq[Double])](
        (p: (DateTime, Seq[Double])) => p._1.isAfter(start) && p._1.isBefore(end)
      )

  val omni_data_path = pwd/'data

  dtfdata.dataset(start.getYear to end.getYear)
    .map(DataPipe((i: Int) => omni_data_path.toString()+"/"+OMNIData.getFilePattern(i)))
    .transform(omni_processing)
    .to_zip(identityPipe[DateTime] * DataPipe((s: Seq[Double]) => Tensor(s).reshape(Shape(s.length))))

}


val scale_dataset = DataPipe((dataset: TFDataSet[(Tensor, Tensor)]) => {

  val concat_features = tfi.stack(
    dataset.training_dataset.map(DataPipe((p: (Tensor, Tensor)) => p._1)).data.toSeq
  ) 

  val concat_targets = tfi.stack(
    dataset.training_dataset.map(DataPipe((p: (Tensor, Tensor)) => p._2)).data.toSeq
  )

  val (min, max) = (concat_targets.min(axes = 0), concat_targets.max(axes = 0))

  val n = concat_features.shape(0)

  val mean_f = concat_features.mean(axes = 0)
  val std_f  = concat_features.subtract(mean_f).square.mean(axes = 0).multiply(n/(n-1)).sqrt
  
  val targets_scaler = MinMaxScalerTF(min, max)

  val features_scaler = GaussianScalerTF(mean_f, std_f)
  
  (
    dataset.copy(
      training_dataset = dataset.training_dataset.map(features_scaler * targets_scaler),
      test_dataset     = dataset.test_dataset.map(features_scaler * identityPipe[Tensor])
    ),
    (features_scaler, targets_scaler)
  )

})


object FTExperiment {
  
  case class Config(
    data_limits: (Int, Int),
    deltaT: (Int, Int),
    log_scale_fte: Boolean
  )


  var config = Config(
    (0, 0),
    (0, 0), 
    false
  )

  var fte_data: ZipDataSet[DateTime, Tensor] = dtfdata.dataset(Iterable[(DateTime, Tensor)]()).to_zip(identityPipe)

  var omni_data: ZipDataSet[DateTime, Tensor] = dtfdata.dataset(Iterable[(DateTime, Tensor)]()).to_zip(identityPipe)

  def clear_cache(): Unit = {
    fte_data = dtfdata.dataset(Iterable[(DateTime, Tensor)]()).to_zip(identityPipe)
    omni_data = dtfdata.dataset(Iterable[(DateTime, Tensor)]()).to_zip(identityPipe)
    config = Config((0, 0), (0, 0), false)
  }

}

val hybrid_poly = (max_degree: Int) => timelagutils.getAct(max_degree, 1)

@main
def apply(
  num_neurons: Seq[Int] = Seq(120, 90),
  activation_func: Int => Activation = hybrid_poly(2),
  optimizer: tf.train.Optimizer = tf.train.Adam(0.001),
  year_range: Range = 2011 to 2017,
  test_year: Int = 2015,
  sw_threshold: Double = 700d,
  deltaT: (Int, Int) = (48, 108),
  reg: Double = 0.0001,
  mo_flag: Boolean = true,
  prob_timelags: Boolean = true,
  log_scale_fte: Boolean = false,
  iterations: Int = 10000,
  miniBatch: Int = 1000,
  fte_data_path: Path = home/'Downloads/'fte) = {


  val sum_dir_prefix = "fte_omni"

  val dt = DateTime.now()

  val summary_dir_index = {
    if(mo_flag) sum_dir_prefix+"_timelag_inference_mo_"+dt.toString("YYYY-MM-dd-HH-mm")
    else sum_dir_prefix+"_timelag_inference_"+dt.toString("YYYY-MM-dd-HH-mm")
  }

  val tf_summary_dir     = home/'tmp/summary_dir_index

  val (test_start, test_end) = (
    new DateTime(test_year, 1, 1, 0, 0),
    new DateTime(test_year, 12, 31, 23, 59)
  )

  val (start, end) = (
    new DateTime(year_range.min, 1, 1, 0, 0),
    new DateTime(year_range.max, 12, 31, 23, 59))


  if(
    FTExperiment.fte_data.size == 0 || 
    FTExperiment.omni_data.size == 0 ||
    FTExperiment.config != FTExperiment.Config((year_range.min, year_range.max), deltaT, log_scale_fte)) {
    
    println("\nProcessing FTE Data")
    FTExperiment.fte_data = load_fte_data(fte_data_path, carrington_rotations, log_scale_fte)(start, end)
    FTExperiment.config = FTExperiment.Config((year_range.min, year_range.max), deltaT, log_scale_fte)

    println("Processing OMNI solar wind data")
    FTExperiment.omni_data = load_solar_wind_data(start, end)(deltaT)

  } else {
    println("\nUsing cached data sets")
  }

  
  val tt_partition = DataPipe((p: (DateTime, (Tensor, Tensor))) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max().scalar.asInstanceOf[Double] >= sw_threshold) false
    else true)

  println("Constructing joined data set")
  val dataset = FTExperiment.fte_data.join(FTExperiment.omni_data).partition(tt_partition)

  val causal_window = dataset.training_dataset.data.head._2._2.shape(0)

  val input_dim = dataset.training_dataset.data.head._2._1.shape(0)

  val input = tf.learn.Input(FLOAT64, Shape(-1, input_dim))

  val trainInput = tf.learn.Input(FLOAT64, Shape(-1, causal_window))

  val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

  val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())


  val num_pred_dims = timelagutils.get_num_output_dims(causal_window, mo_flag, prob_timelags, "default")

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelagutils.get_ffnet_properties(input_dim, num_pred_dims, num_neurons)

  val output_mapping = timelagutils.get_output_mapping(causal_window, mo_flag, prob_timelags, "default", 1.0)

  //Prediction architecture
  val architecture = dtflearn.feedforward_stack(activation_func, FLOAT64)(net_layer_sizes.tail) >> output_mapping


  val lossFunc = timelagutils.get_loss(
    causal_window, mo_flag,
    prob_timelags, 0.0, 1.0,
    0.0, 0.0,
    0.75, "Kullback-Leibler",
    1.0, 1.0, 1.5)

  val loss = lossFunc >>
    L2Regularization(layer_parameter_names, layer_datatypes, layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")


  println("Scaling data attributes")
  val (scaled_data, scalers): helios.data.SC_TF_DATA = scale_dataset(
    dataset.copy(
      training_dataset = dataset.training_dataset.map((p: (DateTime, (Tensor, Tensor))) => p._2),
      test_dataset = dataset.test_dataset.map((p: (DateTime, (Tensor, Tensor))) => p._2)
    )
  )
  
  
  val train_data_tf = {
    scaled_data.training_dataset
      .build[
        (Tensor, Tensor), 
        (Output, Output), 
        (DataType, DataType), 
        (DataType, DataType),
         (Shape, Shape)](
      Left(identityPipe[(Tensor, Tensor)]),
      (FLOAT64, FLOAT64),
      (Shape(input_dim), Shape(causal_window)))
      .repeat()
      .shuffle(10)
      .batch(miniBatch)
      .prefetch(10)
  }

  val (model, estimator) = dtflearn.build_tf_model(
    architecture, input, trainInput, trainingInputLayer,
    loss, optimizer, summariesDir,
    dtflearn.rel_loss_change_stop(0.05, iterations))(
    train_data_tf)


  val nTest = scaled_data.test_dataset.size

  val predictions: (Tensor, Tensor) = dtfutils.buffered_preds[
    Tensor, Output, DataType, Shape, (Output, Output),
    Tensor, Output, DataType, Shape, Output,
    Tensor, (Tensor, Tensor), (Tensor, Tensor)](
    estimator, tfi.stack(scaled_data.test_dataset.data.toSeq.map(_._1), axis = 0),
    500, nTest)

  val index_times = Tensor(
    (0 until causal_window).map(_.toDouble)
  ).reshape(
    Shape(causal_window)
  )

  val pred_time_lags_test: Tensor = if(prob_timelags) {
    val unsc_probs = predictions._2

    unsc_probs.topK(1)._2.reshape(Shape(nTest)).cast(FLOAT64)

  } else predictions._2

  val pred_targets: Tensor = if (mo_flag) {
    val all_preds =
      if (prob_timelags) scalers._2.i(predictions._1)
      else scalers._2.i(predictions._1)

    val repeated_times = tfi.stack(Seq.fill(causal_window)(pred_time_lags_test.floor), axis = -1)

    val conv_kernel = repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

    all_preds.multiply(conv_kernel).sum(axes = 1).divide(conv_kernel.sum(axes = 1))
  } else {
    scalers._2(0).i(predictions._1)
  }

  val test_labels = scaled_data.test_dataset.data.map(_._2).map(t => dtfutils.toDoubleSeq(t).toSeq).toSeq

  val actual_targets = test_labels.zipWithIndex.map(zi => {
    val (z, index) = zi
    val time_lag = pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

    z(time_lag)
  })

  val reg_metrics = new RegressionMetricsTF(pred_targets, actual_targets)


  val experiment_config = helios.ExperimentType(mo_flag, prob_timelags, "mode")

  val results = helios.SupervisedModelRun(
    (scaled_data, scalers),
    model, estimator, None,
    Some(reg_metrics),
    tf_summary_dir, None,
    Some((pred_targets, pred_time_lags_test))
  )

  helios.write_predictions(
    dtfutils.toDoubleSeq(pred_targets).toSeq,
    actual_targets,
    dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
    tf_summary_dir/("scatter_test-"+DateTime.now().toString("YYYY-MM-dd-HH-mm")+".csv"))


  helios.ExperimentResult(
    experiment_config,
    dataset.training_dataset,
    dataset.test_dataset,
    results
  )

}