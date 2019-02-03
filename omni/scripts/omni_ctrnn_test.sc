import org.joda.time._
import ammonite.ops._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.DynaMLPipe._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.utils._
import io.github.mandar2812.dynaml.tensorflow.layers._
import io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._
import io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader, OmniOSA}
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{L2Regularization, L1Regularization}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main(
  yearrange: Range, quantities: Seq[Int] = Seq(Dst, V_SW, B_Z),
  horizon: Int = 24, history: Int = 6, num_hidden_units: Int = 6,
  stop_criteria: StopCriteria = dtflearn.rel_loss_change_stop(0.005, 100000),
  optimizer: Optimizer = tf.train.AdaDelta(0.005),
  reg: Double = 0.001, stormsFile: String = OmniOSA.stormsFileJi) = {

  DateTimeZone.setDefault(DateTimeZone.UTC)

  val target_quantity = OMNIData.columnNames(quantities.head)

  val tf_summary_dir = home/'tmp/("omni_ctrnn_"+target_quantity+"_h-"+history+"_t-"+horizon+"_n-"+num_hidden_units)

  /*
  * Set up the data processing pipeline
  * */

  val process_omni_files = OMNILoader.omniDataToSlidingTS(0, history+horizon+1)(quantities.head, quantities.tail)

  val extract_features_and_targets = StreamDataPipe((d: (DateTime, Seq[Seq[Double]])) => {

    (d._1, d._2.map(s => (s.take(history), s.takeRight(horizon))).unzip)
  })


  val strip_date_stamps = StreamDataPipe((p: (DateTime, (Seq[Seq[Double]], Seq[Seq[Double]]))) => p._2)

  val load_into_tensors = DataPipe((fl: Seq[(Seq[Seq[Double]], Seq[Seq[Double]])]) => {
    val (features, labels) = fl.unzip
    val n_data = features.length
    (
      dtf.tensor_f64(n_data, quantities.length, history)(features.flatMap(_.flatten):_*),
      dtf.tensor_f64(n_data, quantities.length, horizon)(labels.flatMap(_.flatten):_*)
    )
  })

  val data_process_pipe = process_omni_files >
    extract_features_and_targets >
    strip_date_stamps >
    load_into_tensors >
    dtfpipe.gaussian_standardization


  val ((sc_features, sc_labels), scaler) = data_process_pipe(
    yearrange.map(OMNIData.getFilePattern).map("data/"+_).toStream
  )

  /*
  * Set up the model architecture and learning procedure
  * */

  val architecture = tf.learn.Flatten("Flatten_0") >>
    dtflearn.feedforward(num_hidden_units)(0) >>
    dtflearn.Tanh("Tanh_0") >>
    DynamicTimeStepCTRNN("fhctrnn_1", horizon) >>
    FiniteHorizonLinear("fhproj_2", quantities.length)

  val layer_params = Seq(
    "Linear_0/Weights",
    "fhctrnn_1/Weights",
    "fhctrnn_1/Gain",
    "fhproj_2/Weights"
  )

  val layer_shapes = Seq(
    Shape(-1, num_hidden_units),
    Shape(num_hidden_units, num_hidden_units),
    Shape(num_hidden_units, num_hidden_units),
    Shape(quantities.length, num_hidden_units)
  )

  val input = tf.learn.Input(FLOAT64, Shape(-1, quantities.length, history))

  val trainOutput = tf.learn.Input(FLOAT64, Shape(-1, quantities.length, horizon))

  val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT32)

  val lossFunc = MVTimeSeriesLoss("Loss/MVTS")

  val loss = lossFunc >>
    L2Regularization(layer_params, layer_params.map(_ => "FLOAT64"), layer_shapes, reg) >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

  val trainData = tf.data.TensorSlicesDataset(sc_features).zip(tf.data.TensorSlicesDataset(sc_labels))
    .repeat()
    .shuffle(10000)
    .batch(1024)
    .prefetch(10)

  val (model, estimator) = dtflearn.build_tf_model(
    architecture, input, trainOutput, trainingInputLayer,
    loss, optimizer, summariesDir,
    stop_criteria)(
    trainData)
  /*
  * Set up the test data processing pipeline
  * */
  val event_dt_format: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH:mm")

  val readStormsFile = fileToStream > replaceWhiteSpaces

  val getEventDates = StreamDataPipe((line: String) => {
    val splits = line.split(",")

    val start_dt =  event_dt_format.parseDateTime(splits(1)+"/"+splits(2).take(2)+":"+splits(2).takeRight(2))
    val end_dt = event_dt_format.parseDateTime(splits(3)+"/"+splits(4).take(2)+":"+splits(4).takeRight(2))
    (start_dt, end_dt)
  })

  val mapEventsByYear = StreamDataPipe((str_dates: (DateTime, DateTime)) => {

    val (start, end) = str_dates
    val yr = if(start.getYear == end.getYear) Seq(start.getYear) else Seq(start.getYear, end.getYear).sorted
    (yr, str_dates)
  })

  val condenseEvents = DataPipe((str: Stream[(Seq[Int], (DateTime, DateTime))]) => {
    val (years, events) = str.unzip

    val unique_years = years.flatten.distinct.sorted
    (unique_years.map(OMNIData.getFilePattern).map("data/"+_), events)
  })

  val extractDataIntoStream = DataPipe(
    process_omni_files > extract_features_and_targets,
    identityPipe[Stream[(DateTime, DateTime)]])

  val filterDataByEvents = DataPipe2(
    (data: Stream[(DateTime, (Seq[Seq[Double]], Seq[Seq[Double]]))], events: Stream[(DateTime, DateTime)]) => {
      data.filter(p =>
        events.map(e =>
          p._1.isAfter(e._1.minusHours(history)) &&
          p._1.isBefore(e._2.minusHours(horizon))
        ).reduce(_ || _)
      )
    })

  val test_data_pipe = readStormsFile >
    getEventDates >
    mapEventsByYear >
    condenseEvents >
    extractDataIntoStream >
    filterDataByEvents >
    strip_date_stamps >
    load_into_tensors >
    (scaler._1 * identityPipe[Tensor])

  val test_data = test_data_pipe("data/"+stormsFile)

  val metrics = new GenRegressionMetricsTF(scaler._2.i(estimator.infer(() => test_data._1)), test_data._2)

  ((model, estimator), (sc_features, sc_labels), scaler, metrics)
}

def apply(
  yearrange: Range, quantities: Seq[Int] = Seq(Dst, V_SW, B_Z),
  horizon: Int = 24, history: Int = 6, num_hidden_units: Int = 6,
  stop_criteria: StopCriteria = dtflearn.rel_loss_change_stop(0.005, 100000),
  optimizer: Optimizer = tf.train.AdaDelta(0.005),
  reg: Double = 0.001, stormsFile: String = OmniOSA.stormsFileJi) =
  main(
    yearrange, quantities, horizon, history,
    num_hidden_units, stop_criteria, optimizer, reg,
    stormsFile
  )