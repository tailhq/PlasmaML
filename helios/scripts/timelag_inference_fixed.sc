import breeze.numerics._
import breeze.linalg.{DenseMatrix, qr}
import breeze.stats.distributions._
import org.joda.time._
import com.quantifind.charts.Highcharts._
import ammonite.ops._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Layer
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.{DynaMLPipe => Pipe}
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.dynaml.probability.RandomVariable
import _root_.io.github.mandar2812.dynaml.evaluation._
import _root_.io.github.mandar2812.PlasmaML.helios.core._
import _root_.io.github.mandar2812.PlasmaML.helios.data._


//Prediction architecture
val arch = {
  tf.learn.Cast("Input/Cast", FLOAT32) >>
    dtflearn.feedforward(15)(0) >>
    dtflearn.Tanh("SELU_0") >>
    dtflearn.feedforward(2)(1)
}

//Output Function
val output_function =
  DataPipe((v: Tensor) => v.square.sum().sqrt.scalar.asInstanceOf[Float]) >
    DataPipe((x: Float) => x.toDouble)


//Data generation subroutine
def generate_data(
  d: Int, n: Int,
  fixed_lag: Double,
  sliding_window: Int,
  noise: Double,
  noiserot: Double,
  outputPipe: DataPipe[Tensor, Double]) = {

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

  val velocity_pipe = outputPipe > DataPipe((x: Double) => x.toFloat)

  def id[T] = Pipe.identityPipe[T]

  val calculate_outputs =
    velocity_pipe >
      BifurcationPipe(
        DataPipe((_: Float) => fixed_lag),
        id[Float]) >
      DataPipe(DataPipe((l: Double) => l.toInt), id[Float])


  val generate_data_pipe = StreamDataPipe(
    DataPipe(id[Int], BifurcationPipe(id[Tensor], calculate_outputs))  >
      DataPipe((pattern: (Int, (Tensor, (Int, Float)))) =>
        ((pattern._1, pattern._2._1.reshape(Shape(d))), (pattern._1+pattern._2._2._1, pattern._2._2._2)))
  )

  val times = (0 until n).toStream

  val data = generate_data_pipe(times.zip(x))


  val (causes, effects) = data.unzip
  //Create a sliding time window
  val effectsMap = effects.sliding(sliding_window).map(window => (window.head._1, window.map(_._2))).toMap
  //Join the features with sliding time windows of the output
  val joined_data = causes.map(c =>
    if(effectsMap.contains(c._1)) (c._1, (c._2, Some(effectsMap(c._1))))
    else (c._1, (c._2, None)))
    .filter(_._2._2.isDefined)
    .map(p => (p._1, (p._2._1, p._2._2.get)))

  (data, joined_data)

}

@main
def main(
  d: Int = 3, n: Int = 100,
  noise: Double = 0.5, noiserot: Double = 0.1,
  iterations: Int = 100000, fixed_lag: Double = 2d,
  sliding_window: Int = 5,
  outputPipe: DataPipe[Tensor, Double] = output_function,
  architecture: Layer[Output, Output] = arch) = {

  require(sliding_window > fixed_lag, "Forward sliding window must be greater than chosen causal time lag")

  val train_fraction = 0.6

  //Generate synthetic time series data
  val (data, collated_data) = generate_data(
    d, n, fixed_lag, sliding_window,
    noise, noiserot, outputPipe)

  //Transform the generated data into a tensorflow compatible object
  val features = dtf.stack(collated_data.map(_._2._1), axis = 0)

  val labels = dtf.tensor_f32(
    collated_data.length, sliding_window)(
    collated_data.flatMap(_._2._2.map(_.toDouble)):_*)

  val num_training = (collated_data.length*train_fraction).toInt
  val num_test = collated_data.length - num_training

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

  val summary_dir_index = "helios_toy_problem_fixed_"+dt.toString("YYYY-MM-dd-hh-mm")

  val tf_summary_dir = home/'tmp/summary_dir_index

  val input = tf.learn.Input(FLOAT32, Shape(-1, tf_dataset.trainData.shape(1)))

  val num_outputs = sliding_window

  val trainInput = tf.learn.Input(FLOAT32, Shape(-1, num_outputs))

  val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT32)

  val lossFunc = new RBFWeightedSWLoss("Loss/RBFWeightedL1", num_outputs, 1d)

  val loss = lossFunc >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss", "ModelLoss")

  val optimizer = tf.train.AdaDelta(0.005)

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


  val energies = data.map(_._2._2)

  spline(energies)
  title("Output Time Series")

  val effect_times = data.map(_._2._1)

  spline(effect_times)
  hold()
  spline(data.map(_._1._1))
  title("Time Warping/Delay")
  xAxis("Time of Cause, t")
  yAxis("Time of Effect, "+0x03C6.toChar+"(t)")
  legend(Seq("t_ = "+0x03C6.toChar+"(t)", "t_ = t"))
  unhold()

  val predictions = estimator.infer(() => tf_dataset.testData)

  val pred_targets = predictions(::, 0)
    .multiply(labels_stddev(0))
    .add(labels_mean(0))

  val pred_time_lags = predictions(::, 1)

  val metrics = new HeliosOmniTSMetrics(
    dtf.stack(Seq(pred_targets, pred_time_lags), axis = 1), tf_dataset.testLabels,
    tf_dataset.testLabels.shape(1),
    dtf.tensor_f32(tf_dataset.nTest)(Seq.fill(tf_dataset.nTest)(lossFunc.time_scale):_*)
  )

  val mae_lag = pred_time_lags
    .sigmoid
    .multiply(num_outputs-1)
    .subtract(fixed_lag)
    .abs.mean()
    .scalar
    .asInstanceOf[Double]

  print("Mean Absolute Error in time lag = ")
  pprint.pprintln(mae_lag)

  val reg_metrics = new RegressionMetricsTF(pred_targets, tf_dataset.testLabels(::, fixed_lag.toInt))

  histogram(pred_time_lags.sigmoid.multiply(num_outputs-1).entriesIterator.map(_.asInstanceOf[Float]).toSeq)
  title("Predicted Time Lags")

  (collated_data, tf_dataset, model, estimator, tf_summary_dir, metrics, reg_metrics)
}
