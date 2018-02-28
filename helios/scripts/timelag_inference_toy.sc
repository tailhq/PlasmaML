import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.{DynaMLPipe => Pipe}
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.dynaml.probability.RandomVariable
import _root_.io.github.mandar2812.dynaml.evaluation._
import breeze.linalg.{DenseMatrix, DenseVector, qr}
import breeze.stats.distributions.Gaussian
import com.quantifind.charts.Highcharts._
import _root_.io.github.mandar2812.PlasmaML.helios.data.HeliosDataSet
import ammonite.ops.home
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.PlasmaML.helios.core._
import org.platanios.tensorflow.api.learn.layers.Layer

//Prediction architecture
val arch = {
  tf.learn.Cast("Input/Cast", FLOAT32) >>
    dtflearn.feedforward(20)(0) >>
    dtflearn.Tanh("Tanh_0") >>
    dtflearn.feedforward(2)(1)
}

def generate_data(
  d: Int = 3, n: Int = 5,
  sliding_window: Int,
  noise: Double = 0.5,
  noiserot: Double = 0.1) = {

  val random_gaussian_vec = DataPipe((i: Int) => RandomVariable(
    () => dtf.tensor_f32(i, 1)((0 until i).map(_ => scala.util.Random.nextGaussian()*noise):_*)
  ))

  val normalise = DataPipe((t: RandomVariable[Tensor]) => t.draw.l2Normalize(0))

  val normalised_gaussian_vec = random_gaussian_vec > normalise

  val x0 = normalised_gaussian_vec(d)

  val random_gaussian_mat = DataPipe(
    (n: Int) => DenseMatrix.rand(n, n, Gaussian(0d, noiserot))
  )

  val rand_rot_mat =
    random_gaussian_mat >
      DataPipe((m: DenseMatrix[Double]) => qr(m).q) >
      DataPipe((m: DenseMatrix[Double]) => dtf.tensor_f32(m.rows, m.rows)(m.toArray:_*).transpose())


  val rotation = rand_rot_mat(d)

  val get_rotation_operator = MetaPipe((rotation_mat: Tensor) => (x: Tensor) => rotation_mat.matmul(x))

  val rotation_op = get_rotation_operator(rotation)

  val translation_op = DataPipe2((tr: Tensor, x: Tensor) => tr.add(x))

  val translation_vecs = random_gaussian_vec(d).iid(n-1).draw

  val x_tail = translation_vecs.scanLeft(x0)((x, sc) => translation_op(sc, rotation_op(x)))

  val x: Seq[Tensor] = Stream(x0) ++ x_tail

  val velocity_pipe = DataPipe((v: Tensor) => v.square.sum().sqrt.scalar.asInstanceOf[Float])

  def id[T] = Pipe.identityPipe[T]

  val calculate_outputs =
    velocity_pipe >
      BifurcationPipe(
        DataPipe((v: Float) => 10/(v+ 1E-6)),
        id[Float]) >
      DataPipe(DataPipe((d: Double) => d.toInt), id[Float])


  val generate_data_pipe = StreamDataPipe(
    DataPipe(id[Int], BifurcationPipe(id[Tensor], calculate_outputs))  >
      DataPipe((pattern: (Int, (Tensor, (Int, Float)))) =>
        ((pattern._1, pattern._2._1.reshape(Shape(d))), (pattern._1+pattern._2._2._1, pattern._2._2._2)))
  )

  val times = (0 until n).toStream

  val data = generate_data_pipe(times.zip(x))

  val (causes, effects) = data.unzip

  val outputs = effects.groupBy(_._1).mapValues(v => v.map(_._2).sum/v.length.toDouble).toSeq.sortBy(_._1)

  val linear_segments = outputs.sliding(2).toList.map(s =>
    DataPipe((t: Double) => {

      val (tmin, tmax) = (s.head._1.toDouble, s.last._1.toDouble)
      val (v0, v1) = (s.head._2, s.last._2)
      val slope: Double = (v1 - v0)/(tmax - tmin)

      if(t >= tmin && t < tmax) v0 + slope*(t - tmin)
      else 0d
    })
  )

  val interpolated_output_signal = causes.map(_._1).map(t => (t, linear_segments.map(_.run(t.toDouble)).sum))

  val effectsMap = interpolated_output_signal
    .sliding(sliding_window)
    .map(window => (window.head._1, window.map(_._2)))
    .toMap

  //Join the features with sliding time windows of the output
  val joined_data = data.map(c =>
    if(effectsMap.contains(c._1._1)) (c._1._1, (c._1._2, Some(effectsMap(c._1._1)), c._2._1 - c._1._1))
    else (c._1._1, (c._1._2, None, c._2._1 - c._1._1)))
    .filter(_._2._2.isDefined)
    .map(p => (p._1, (p._2._1, p._2._2.get, p._2._3)))

  (data, joined_data)
}

@main
def main(
  d: Int = 3, n: Int = 100,
  sliding_window: Int = 15,
  noise: Double = 0.5,
  noiserot: Double = 0.1,
  iterations: Int = 150000,
  architecture: Layer[Output, Output] = arch) = {

  val train_fraction = 0.7

  val (data, collated_data) = generate_data(d, n, sliding_window, noise, noiserot)

  val (causes, effects) = data.unzip

  val energies = data.map(_._2._2)

  spline(energies)
  title("Energy Time Series")

  val effect_times = data.map(_._2._1)

  histogram(effects.map(_._2), numBins = 100)
  title("Distribution of output signal  ")

  histogram(effect_times.zip(causes.map(_._1)).map(c => c._1 - c._2), numBins = 10)
  title("Distribution of time lags")

  spline(effect_times)
  hold()
  spline(data.map(_._1._1))
  title("Time Warping/Delay")
  xAxis("Time of Cause, t")
  yAxis("Time of Effect, "+0x03C6.toChar+"(t)")
  legend(Seq("t_ = "+0x03C6.toChar+"(t)", "t_ = t"))
  unhold()

  val outputs = effects.groupBy(_._1).mapValues(v => v.map(_._2).sum/v.length.toDouble).toSeq.sortBy(_._1)

  line(outputs)
  hold()
  line(energies)
  legend(Seq("Output Data with Lag", "Output Data without Lag"))
  unhold()


  //Transform the generated data into a tensorflow compatible object
  val features = dtf.stack(collated_data.map(_._2._1), axis = 0)

  val labels = dtf.tensor_f32(
    collated_data.length, sliding_window)(
    collated_data.flatMap(_._2._2):_*)

  val labels_timelags = dtf.tensor_f32(collated_data.length)(collated_data.map(d => d._2._3.toDouble):_*)

  val num_training = (collated_data.length*train_fraction).toInt
  val num_test = collated_data.length - num_training


  val (training_time_lags, test_time_lags): (Tensor, Tensor) = (
    labels_timelags(0 :: num_training),
    labels_timelags(num_training :: ))


  //Create a helios data set.
  val tf_dataset = HeliosDataSet(
    features(0 :: num_training, ---), labels(0 :: num_training), num_training,
    features(num_training ::, ---), labels(num_training ::), num_test)

  val labels_mean = tf_dataset.trainLabels.mean(axes = Tensor(0))

  val labels_stddev = tf_dataset.trainLabels.subtract(labels_mean).square.mean(axes = Tensor(0)).sqrt

  val norm_train_labels = tf_dataset.trainLabels.subtract(labels_mean).divide(labels_stddev)

  val training_data = tf.data.TensorSlicesDataset(tf_dataset.trainData)
    .zip(tf.data.TensorSlicesDataset(norm_train_labels)).repeat()
    .shuffle(100)
    .batch(256)
    .prefetch(10)

  val dt = DateTime.now()

  val summary_dir_index = "helios_toy_problem_variable_"+dt.toString("YYYY-MM-dd-hh-mm")

  val tf_summary_dir = home/'tmp/summary_dir_index

  val input = tf.learn.Input(FLOAT32, Shape(-1, tf_dataset.trainData.shape(1)))

  val num_outputs = sliding_window

  val trainInput = tf.learn.Input(FLOAT32, Shape(-1, num_outputs))

  val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT32)

  val lossFunc = new RBFWeightedSWLoss("Loss/RBFWeightedL1", num_outputs, 1d)

  val loss = lossFunc >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val optimizer = tf.train.AdaDelta(0.01)

  val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

  val (model, estimator) = tf.createWith(graph = Graph()) {
    val model = tf.learn.Model(
      input, architecture, trainInput, trainingInputLayer,
      loss, optimizer)

    println("Training the regression model.")

    val estimator = tf.learn.FileBasedEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(iterations)),
      Set(
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(100))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 100))

    estimator.train(() => training_data, tf.learn.StopCriteria(maxSteps = Some(iterations)))

    (model, estimator)
  }

  val predictions = estimator.infer(() => tf_dataset.testData)

  val pred_targets = predictions(::, 0)
    .multiply(labels_stddev(0))
    .add(labels_mean(0))

  val unscaled_pred_time_lags_test = predictions(::, 1)

  val metrics = new HeliosOmniTSMetrics(
    dtf.stack(Seq(pred_targets, unscaled_pred_time_lags_test), axis = 1), tf_dataset.testLabels,
    tf_dataset.testLabels.shape(1),
    dtf.tensor_f32(tf_dataset.nTest)(Seq.fill(tf_dataset.nTest)(lossFunc.time_scale):_*)
  )

  val pred_time_lags_test = unscaled_pred_time_lags_test
    .sigmoid
    .multiply(num_outputs-1)

  val err_time_lag_test = pred_time_lags_test.subtract(test_time_lags)

  val reg_time_lag = new RegressionMetricsTF(pred_time_lags_test, test_time_lags)

  val mae_lag = err_time_lag_test
    .abs.mean()
    .scalar
    .asInstanceOf[Float]

  print("Mean Absolute Error in time lag = ")
  pprint.pprintln(mae_lag)

  val actual_targets = (0 until num_test).toSeq.map(n => {
    val time_lag = pred_time_lags_test(n).scalar.asInstanceOf[Float].toInt
    tf_dataset.testLabels(n, time_lag).scalar.asInstanceOf[Float]
  })

  val reg_metrics = new RegressionMetricsTF(pred_targets, actual_targets)

  histogram(unscaled_pred_time_lags_test.sigmoid.multiply(num_outputs-1).entriesIterator.map(_.asInstanceOf[Float]).toSeq)
  title("Predicted Time Lags")

  histogram(err_time_lag_test.entriesIterator.toSeq.map(_.asInstanceOf[Float]), numBins = 100)
  title("Histogram of Time Lag prediction errors")
  unhold()

  val test_signal_predicted = collated_data.slice(num_training, n).zipWithIndex.map(c => {
    val time_index = c._1._1
    val pred_lag = pred_time_lags_test(c._2).scalar.asInstanceOf[Float]
    val pred = pred_targets(c._2).scalar.asInstanceOf[Float]
    (time_index + pred_lag, pred)
  }).sortBy(_._1)


  line(outputs.slice(num_training, n))
  hold()
  line(test_signal_predicted)
  legend(Seq("Actual Output Signal", "Predicted Output Signal"))
  title("Test Set Predictions")
  unhold()

  (collated_data, tf_dataset, model, estimator, tf_summary_dir, metrics, reg_metrics, reg_time_lag)

}
