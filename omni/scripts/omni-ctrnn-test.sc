import _root_.io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.PlasmaML.dynamics.nn.FiniteHorizonCTRNN
import org.joda.time._
import ammonite.ops._
import io.github.mandar2812.PlasmaML.utils.MVTimeSeriesLoss
import io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._
import io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

@main
def main(
  yearrange: Range, quantities: Seq[Int] = Seq(Dst, V_SW, B_Z),
  horizon: Int = 24, iterations: Int = 50000) = {

  val target_quantity = OMNIData.columnNames(quantities.head)

  val tf_summary_dir = home/'tmp/("omni_ctrnn_"+target_quantity+"_horizon-"+horizon)

  val process_omni_files = OMNILoader.omniDataToSlidingTS(0, horizon+1)(quantities.head, quantities.tail)

  val extract_features_and_targets = StreamDataPipe((d: (DateTime, Seq[Seq[Double]])) => {

    (d._1, d._2.map(s => (s.head, s.tail)).unzip)
  })


  val strip_date_stamps = StreamDataPipe((p: (DateTime, (Seq[Double], Seq[Seq[Double]]))) => p._2)

  val load_into_tensors = DataPipe((fl: Seq[(Seq[Double], Seq[Seq[Double]])]) => {
    val (features, labels) = fl.unzip
    val n_data = features.length
    (
      dtf.tensor_f32(n_data, quantities.length)(features.flatten:_*),
      dtf.tensor_f32(n_data, quantities.length, horizon)(labels.flatMap(_.flatten):_*)
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

  val architecture = FiniteHorizonCTRNN("fhctrnn_0", quantities.length, horizon, 1d)

  val input = tf.learn.Input(FLOAT64, Shape(-1, quantities.length))

  val trainOutput = tf.learn.Input(FLOAT64, Shape(-1, quantities.length, horizon))

  val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT32)

  val lossFunc = MVTimeSeriesLoss("Loss/MVTS")

  val loss = lossFunc >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

  val optimizer = tf.train.AdaDelta(0.005)

  val trainData = tf.data.TensorSlicesDataset(sc_features).zip(tf.data.TensorSlicesDataset(sc_labels))
    .repeat()
    .shuffle(10000)
    .batch(256)
    .prefetch(10)

  val (model, estimator) = tf.createWith(graph = Graph()) {
    val model = tf.learn.Model(
      input, architecture, trainOutput, trainingInputLayer,
      loss, optimizer)

    println("Training the linear regression model.")

    val estimator = tf.learn.FileBasedEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(iterations)),
      Set(
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(5000)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(5000)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(5000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 5000))

    estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(iterations)))

    (model, estimator)
  }

  ((model, estimator), (sc_features, sc_labels), scaler)
}

def apply(
  yearrange: Range, quantities: Seq[Int] = Seq(Dst, V_SW, B_Z),
  horizon: Int = 24, iterations: Int = 50000) =
  main(yearrange, quantities, horizon, iterations)