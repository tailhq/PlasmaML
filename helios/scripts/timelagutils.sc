import breeze.linalg.{DenseMatrix, qr}
import breeze.stats.distributions.Gaussian
import com.quantifind.charts.Highcharts._
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.learn.layers.{Activation, Input, Layer, Loss}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.io.data.Dataset
import _root_.io.github.mandar2812.dynaml.{DynaMLPipe => Pipe}
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.utils._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.probability.RandomVariable
import _root_.io.github.mandar2812.dynaml.evaluation._
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core._
import _root_.io.github.mandar2812.PlasmaML.helios.data.HeliosDataSet
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.{Mode, SupervisedTrainableModel}

//Define some types for convenience.
type DATA        = Stream[((Int, Tensor), (Float, Float))]
type SLIDINGDATA = Stream[(Int, (Tensor, Stream[Double], Float))]
type TLDATA      = (DATA, SLIDINGDATA)

//Alias for the identity pipe/mapping
def id[T]: DataPipe[T, T] = Pipe.identityPipe[T]

//A Polynomial layer builder
val layer_poly = (power: Int) => (n: String) => new Activation(n) {
  override val layerType = "Poly"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    //val power = dtf.tensor_i32(input.shape(1))((1 to input.shape(1)):_*)

    input.pow(power)

  }
}

val getAct = (s: Int) => (i: Int) =>
  if(i - s == 0) layer_poly(2)(s"Act_$i")
  else tf.learn.Sigmoid(s"Act_$i")

//subroutine to calculate sliding autocorrelation of a time series.
def autocorrelation(n: Int)(data: Stream[Double]): Stream[Double] = {
  val mean = data.sum/data.length
  val variance = data.map(_ - mean).map(math.pow(_, 2d)).sum/(data.length - 1d)


  (0 to n).map(lag => {
    val sliding_ts = data.sliding(lag+1).toSeq
    val len = sliding_ts.length - 1d

    sliding_ts.map(xs => (xs.head - mean) * (xs.last - mean)).sum/(len*variance)
  }).toStream
}

//Subroutine to generate synthetic
//input-lagged output time series.
def generate_data(
  d: Int = 3, n: Int = 5,
  sliding_window: Int,
  noise: Double = 0.5,
  noiserot: Double = 0.1,
  alpha: Double = 0.0,
  compute_output_and_lag: DataPipe[Tensor, (Float, Float)]): TLDATA = {

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

  val translation_op = DataPipe2((tr: Tensor, x: Tensor) => tr.add(x.multiply(1.0f - alpha.toFloat)))

  val translation_vecs = random_gaussian_vec(d).iid(n+500-1).draw

  val x_tail = translation_vecs.scanLeft(x0)((x, sc) => translation_op(sc, rotation_op(x)))

  val x: Seq[Tensor] = (Stream(x0) ++ x_tail).takeRight(n)

  val calculate_outputs =
    compute_output_and_lag >
      DataPipe(
        DataPipe((d: Float) => d),
        DataPipe((v: Float) => v)
      )


  val generate_data_pipe = StreamDataPipe(
    DataPipe(id[Int], BifurcationPipe(id[Tensor], calculate_outputs))  >
      DataPipe((pattern: (Int, (Tensor, (Float, Float)))) =>
        ((pattern._1, pattern._2._1.reshape(Shape(d))), (pattern._1+pattern._2._2._1, pattern._2._2._2)))
  )

  val times = (0 until n).toStream

  val data = generate_data_pipe(times.zip(x))

  val (causes, effects) = data.unzip

  val outputs = effects.groupBy(_._1.toInt).mapValues(v => v.map(_._2).sum/v.length.toDouble).toSeq.sortBy(_._1)

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


  val energies = data.map(_._2._2)

  spline(energies)
  title("Output Time Series")

  val effect_times = data.map(_._2._1)

  try {

    histogram(effect_times.zip(causes.map(_._1)).map(c => c._1 - c._2))
    title("Distribution of time lags")

  } catch {
    case _: java.util.NoSuchElementException => println("Can't plot histogram due to `No Such Element` exception")
    case _ => println("Can't plot histogram due to exception")
  }

  line(outputs)
  hold()
  line(energies)
  legend(Seq("Output Data with Lag", "Output Data without Lag"))
  unhold()

  spline(autocorrelation(2*sliding_window)(data.map(_._2._2.toDouble)))
  title("Auto-covariance of time series")

  (data, joined_data)
}

//Transform the generated data into a tensorflow compatible object
def load_data_into_tensors(num_training: Int, num_test: Int, sliding_window: Int) =
  DataPipe((data: SLIDINGDATA) => {
    require(

      num_training + num_test == data.length,
      "Size of train and test data must add up to total size of data!")

    (data.take(num_training), data.takeRight(num_test))
  }) > data_splits_to_tensors(sliding_window)

def data_splits_to_tensors(sliding_window: Int) =
  DataPipe2((training_data: SLIDINGDATA, test_data: SLIDINGDATA)=> {

    val features_train = dtf.stack(training_data.map(_._2._1), axis = 0)

    val features_test  = dtf.stack(test_data.map(_._2._1), axis = 0)

    val labels_tr_flat = training_data.toList.flatMap(_._2._2.toList)

    val labels_train = dtf.tensor_f64(
      training_data.length, sliding_window)(
      labels_tr_flat:_*)

    val labels_te_flat = test_data.toList.flatMap(_._2._2.toList)

    val labels_test  = dtf.tensor_f64(
      test_data.length, sliding_window)(
      labels_te_flat:_*)

    val (train_time_lags, test_time_lags): (Tensor, Tensor) = (
      dtf.tensor_f64(training_data.length)(training_data.toList.map(d => d._2._3.toDouble):_*),
      dtf.tensor_f64(test_data.length)(test_data.toList.map(d => d._2._3.toDouble):_*))


    //Create a helios data set.
    val tf_dataset = HeliosDataSet(
      features_train, labels_train, training_data.length,
      features_test, labels_test, test_data.length)

    (tf_dataset, (train_time_lags, test_time_lags))
  })


//Scale training features/labels, apply scaling to test features

val scale_helios_dataset = DataPipe((dataset: HeliosDataSet) => {

  val (norm_tr_data, scalers) = dtfpipe.gaussian_standardization(dataset.trainData, dataset.trainLabels)

  (
    dataset.copy(
      trainData = norm_tr_data._1, trainLabels = norm_tr_data._2,
      testData = scalers._1(dataset.testData)),
    scalers
  )
})

val scale_data = DataPipe(
  scale_helios_dataset,
  id[(Tensor, Tensor)]
)

def get_ffnet_properties(
  d: Int, num_pred_dims: Int,
  num_neurons: Int, num_hidden_layers: Int) = {

  val net_layer_sizes       = Seq(d) ++ Seq.fill(num_hidden_layers)(num_neurons) ++ Seq(num_pred_dims)
  val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))
  val layer_parameter_names = (1 to net_layer_sizes.tail.length).map(s => "Linear_"+s+"/Weights")
  val layer_datatypes       = Seq.fill(net_layer_sizes.tail.length)("FLOAT64")

  (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes)
}

def get_ffnet_properties(
  d: Int, num_pred_dims: Int,
  layer_sizes: Seq[Int]) = {

  val net_layer_sizes       = Seq(d) ++ layer_sizes ++ Seq(num_pred_dims)
  val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))
  val layer_parameter_names = (1 to net_layer_sizes.tail.length).map(s => "Linear_"+s+"/Weights")
  val layer_datatypes       = Seq.fill(net_layer_sizes.tail.length)("FLOAT64")

  (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes)
}

def get_output_mapping(
  causal_window: Int,
  mo_flag: Boolean,
  prob_timelags: Boolean,
  dist_type: String,
  time_scale: Double = 1.0) = if (!mo_flag) {
  if (!prob_timelags) RBFWeightedSWLoss.output_mapping("Output/RBFWeightedL1", causal_window, time_scale)
  else CausalDynamicTimeLagSO.output_mapping("Output/SOProbWeightedTS", causal_window)

} else if(mo_flag && !prob_timelags) {

  MOGrangerLoss.output_mapping("Output/MOGranger", causal_window)

} else {
  dist_type match {
    case "poisson"  => helios.learn.cdt_poisson_loss.output_mapping("Output/PoissonWeightedTS", causal_window)
    case "beta"     => helios.learn.cdt_beta_loss.output_mapping("Output/BetaWeightedTS", causal_window)
    case "gaussian" => helios.learn.cdt_gaussian_loss.output_mapping("Output/GaussianWeightedTS", causal_window)
    case _          => helios.learn.cdt_loss.output_mapping("Output/ProbWeightedTS", causal_window)
  }
}

def get_num_output_dims(
  causal_window: Int,
  mo_flag: Boolean,
  prob_timelags: Boolean,
  dist_type: String) =
  if(!mo_flag) 2
  else if(mo_flag && !prob_timelags) causal_window + 1
  else if(!mo_flag && prob_timelags) causal_window + 1
  else {
    dist_type match {
      case "poisson"  => causal_window + 1
      case "gaussian" => causal_window + 2
      case "beta"     => causal_window + 2
      case _          => 2*causal_window
    }
  }


def get_loss(
  sliding_window: Int,
  mo_flag: Boolean,
  prob_timelags: Boolean,
  p: Double                     = 1.0,
  time_scale: Double            = 1.0,
  corr_sc: Double               = 2.5,
  c_cutoff: Double              = 0.0,
  prior_wt: Double              = 1d,
  prior_divergence: String      = "Hellinger",
  temp: Double                  = 1.0,
  error_wt: Double              = 1.0,
  c: Double                     = 1.0) =
  if (!mo_flag) {
    if (!prob_timelags) {
      RBFWeightedSWLoss(
        "Loss/RBFWeightedL1", sliding_window,
        kernel_time_scale = time_scale,
        kernel_norm_exponent = p,
        corr_cutoff = c_cutoff,
        prior_scaling = corr_sc,
        batch = 512)
    } else {
      helios.learn.cdt_loss_so(
        "Loss/ProbWeightedTS",
        sliding_window,
        prior_wt = prior_wt,
        temperature = 0.75,
        divergence = prior_divergence,
        specificity = c)
    }

  } else if(mo_flag && !prob_timelags) {

    MOGrangerLoss(
      "Loss/MOGranger", sliding_window,
      error_exponent = p,
      weight_error = prior_wt)

  } else {
    helios.learn.cdt_loss(
      "Loss/ProbWeightedTS",
      sliding_window,
      prior_wt = prior_wt,
      error_wt = error_wt,
      temperature = temp,
      divergence = prior_divergence,
      specificity = c)
  }

def process_predictions(
  predictions: (Tensor, Tensor),
  time_window: Int,
  multi_output: Boolean = true,
  probabilistic_time_lags: Boolean = true,
  timelag_pred_strategy: String = "mode",
  scale_outputs: Option[GaussianScalerTF] = None): (Tensor, Tensor) = {

  val index_times = Tensor(
    (0 until time_window).map(_.toDouble)
  ).reshape(
    Shape(time_window)
  )

  val pred_time_lags = if(probabilistic_time_lags) {
    val unsc_probs = predictions._2

    if (timelag_pred_strategy == "mode") unsc_probs.topK(1)._2.reshape(Shape(predictions._1.shape(0))).cast(FLOAT64)
    else unsc_probs.multiply(index_times).sum(axes = 1)

  } else predictions._2

  val pred_targets: Tensor = if (multi_output) {

    val all_preds =
      if (scale_outputs.isDefined) scale_outputs.get.i(predictions._1)
      else predictions._1

    val repeated_times = tfi.stack(Seq.fill(time_window)(pred_time_lags.floor), axis = -1)

    val conv_kernel = repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

    all_preds.multiply(conv_kernel).sum(axes = 1).divide(conv_kernel.sum(axes = 1))

  } else {

    if (scale_outputs.isDefined) {
      val scaler = scale_outputs.get
      scaler(0).i(predictions._1)
    } else predictions._1

  }

  (pred_targets, pred_time_lags)

}

def plot_time_series(targets: Tensor, predictions: Tensor, plot_title: String): Unit = {
  line(dtfutils.toDoubleSeq(targets).zipWithIndex.map(c => (c._2, c._1)).toSeq)
  hold()
  line(dtfutils.toDoubleSeq(predictions).zipWithIndex.map(c => (c._2, c._1)).toSeq)
  legend(Seq("Actual Output Signal", "Predicted Output Signal"))
  title(plot_title)
  unhold()
}

def plot_time_series(targets: Stream[(Int, Double)], predictions: Tensor, plot_title: String): Unit = {
  line(targets.toSeq)
  hold()
  line(dtfutils.toDoubleSeq(predictions).zipWithIndex.map(c => (c._2, c._1)).toSeq)
  legend(Seq("Actual Output Signal", "Predicted Output Signal"))
  title(plot_title)
  unhold()
}

@throws[java.util.NoSuchElementException]
@throws[Exception]
def plot_histogram(data: Tensor, plot_title: String): Unit = {
  try {

    histogram(dtfutils.toDoubleSeq(data).toSeq)
    title(plot_title)

  } catch {
    case _: java.util.NoSuchElementException => println("Can't plot histogram due to `No Such Element` exception")
    case _ => println("Can't plot histogram due to exception")
  }
}

def plot_input_output(
  input: Stream[Tensor],
  input_to_scalar: Tensor => Double,
  targets: Tensor,
  predictions: Tensor,
  xlab: String,
  ylab: String,
  plot_title: String): Unit = {

  val processed_inputs = input.map(input_to_scalar)

  scatter(processed_inputs.zip(dtfutils.toDoubleSeq(predictions).toSeq))

  hold()
  scatter(processed_inputs.zip(dtfutils.toDoubleSeq(targets).toSeq))
  xAxis(xlab)
  yAxis(ylab)
  title(plot_title)
  legend(Seq("Model", "Data"))
  unhold()
}

def plot_scatter(
  x: Tensor,
  y: Tensor,
  xlab: Option[String] = None,
  ylab: Option[String] = None,
  plot_title: Option[String] = None): Seq[(Double, Double)] = {
  val xy = dtfutils.toDoubleSeq(x).zip(dtfutils.toDoubleSeq(y)).toSeq
  scatter(xy)
  if(xlab.isDefined) xAxis(xlab.get)
  if(ylab.isDefined) yAxis(ylab.get)
  if(plot_title.isDefined) title(plot_title.get)
  xy
}

sealed trait ModelRun {

  type MODEL
  type ESTIMATOR

  val summary_dir: Path
  val timelags_gt: (Tensor, Tensor)
  val data_and_scales: (HeliosDataSet, (GaussianScalerTF, GaussianScalerTF))
  val metrics: (RegressionMetricsTF, RegressionMetricsTF)
  val model: MODEL
  val estimator: ESTIMATOR

}

case class JointModelRun(
  data_and_scales: (HeliosDataSet, (GaussianScalerTF, GaussianScalerTF)),
  model: SupervisedTrainableModel[
    Tensor, Output, DataType, Shape, (Output, Output),
    Tensor, Output, DataType, Shape, Output],
  estimator: Estimator[
    Tensor, Output, DataType, Shape, (Output, Output),
    (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
    ((Output, Output), Output)],
  metrics: (RegressionMetricsTF, RegressionMetricsTF),
  summary_dir: Path,
  timelags_gt: (Tensor, Tensor)) extends ModelRun {

  override type MODEL = SupervisedTrainableModel[
    Tensor, Output, DataType, Shape, (Output, Output),
    Tensor, Output, DataType, Shape, Output]

  override type ESTIMATOR = Estimator[
    Tensor, Output, DataType, Shape, (Output, Output),
    (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
    ((Output, Output), Output)]
}

case class StageWiseModelRun(
  data_and_scales: (HeliosDataSet, (GaussianScalerTF, GaussianScalerTF)),
  model: SupervisedTrainableModel[
    Tensor, Output, DataType, Shape, (Output, Output),
    Tensor, Output, DataType, Shape, Output],
  estimator: Estimator[
    Tensor, Output, DataType, Shape, (Output, Output),
    (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
    ((Output, Output), Output)],
  metrics: (RegressionMetricsTF, RegressionMetricsTF),
  summary_dir: Path,
  timelags_gt: (Tensor, Tensor)) extends ModelRun {

  override type MODEL = SupervisedTrainableModel[
    Tensor, Output, DataType, Shape, (Output, Output),
    Tensor, Output, DataType, Shape, Output]

  override type ESTIMATOR = Estimator[
    Tensor, Output, DataType, Shape, (Output, Output),
    (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
    ((Output, Output), Output)]
}


case class ExperimentType(
  multi_output: Boolean,
  probabilistic_time_lags: Boolean,
  timelag_prediction: String)

case class ExperimentResult(
  config: ExperimentType,
  train_data: TLDATA,
  test_data: TLDATA,
  results: JointModelRun
)

def plot_and_write_results(results: ExperimentResult): Unit = {

  val ExperimentResult(
  ExperimentType(mo_flag, prob_timelags, timelag_pred_strategy),
  (_, collated_data),
  (_, collated_data_test),
  JointModelRun(
  (tf_dataset, scalers),
  _, estimator,
  (reg_metrics, reg_time_lag),
  tf_summary_dir,
  (train_time_lags, _))) = results

  val err_time_lag_test = reg_time_lag.preds.subtract(reg_time_lag.targets)

  val mae_lag = err_time_lag_test
    .abs.mean()
    .scalar
    .asInstanceOf[Double]

  val pred_time_lags_test = reg_time_lag.preds

  print("Mean Absolute Error in time lag = ")
  pprint.pprintln(mae_lag)

  plot_histogram(pred_time_lags_test, plot_title = "Predicted Time Lags")
  plot_histogram(err_time_lag_test, plot_title = "Time Lag prediction errors")

  plot_time_series(reg_metrics.targets, reg_metrics.preds, plot_title = "Test Set Predictions")

  plot_input_output(
    input = collated_data_test.map(_._2._1),
    input_to_scalar = (t: Tensor) => t.abs.sum().scalar.asInstanceOf[Float].toDouble,
    targets = reg_metrics.targets,
    predictions = reg_metrics.preds,
    xlab = "||x(t)||_1",
    ylab = "f(x(t))",
    plot_title = "Input-Output Relationship: Test Data"
  )

  //Perform same visualisation for training set
  val training_preds: (Tensor, Tensor) = estimator.infer(() => tf_dataset.trainData)

  val (train_signal_predicted, pred_time_lags_train) = process_predictions(
    training_preds,
    collated_data.head._2._2.length,
    mo_flag,
    prob_timelags,
    timelag_pred_strategy,
    Some(scalers._2))


  val unscaled_train_labels = scalers._2.i(tf_dataset.trainLabels)

  val training_signal_actual = (0 until tf_dataset.nTrain).map(n => {
    val time_lag = pred_time_lags_train(n).scalar.asInstanceOf[Double].toInt
    unscaled_train_labels(n, time_lag).scalar.asInstanceOf[Double]
  })

  plot_time_series(
    collated_data.map(c => (c._1+c._2._3.toInt, c._2._2(c._2._3.toInt))),
    train_signal_predicted,
    plot_title = "Training Set Predictions")

  plot_input_output(
    input = collated_data.map(_._2._1),
    input_to_scalar = (t: Tensor) => t.abs.sum().scalar.asInstanceOf[Float].toDouble,
    targets = training_signal_actual,
    predictions = train_signal_predicted,
    xlab = "||x(t)||_1",
    ylab = "f(x(t))",
    plot_title = "Input-Output Relationship: Training Data"
  )


  val err_train     = train_signal_predicted.subtract(training_signal_actual)
  val err_lag_train = pred_time_lags_train.subtract(train_time_lags)

  val train_err_scatter = plot_scatter(
    err_train,
    err_lag_train,
    xlab = Some("Error in Velocity"),
    ylab = Some("Error in Time Lag"),
    plot_title = Some("Training Set Errors; Scatter")
  )

  write(
    tf_summary_dir/"train_errors.csv",
    "error_v,error_lag\n"+train_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))


  val train_scatter = plot_scatter(
    train_signal_predicted,
    pred_time_lags_train,
    xlab = Some("Velocity"),
    ylab = Some("Time Lag"),
    plot_title = Some("Training Set; Scatter"))

  hold()

  val train_actual_scatter = plot_scatter(
    training_signal_actual,
    train_time_lags)
  legend(Seq("Predictions", "Actual Data"))
  unhold()

  write(
    tf_summary_dir/"train_scatter.csv",
    "predv,predlag,actualv,actuallag\n"+
      train_scatter.zip(train_actual_scatter).map(
        c => c._1._1.toString + "," + c._1._2.toString + "," + c._2._1.toString + "," + c._2._2.toString
      ).mkString("\n")
  )

  val err_test     = reg_metrics.preds.subtract(reg_metrics.targets)
  val err_lag_test = reg_time_lag.preds.subtract(reg_time_lag.targets)

  val test_err_scatter = plot_scatter(
    err_test, err_lag_test,
    xlab = Some("Error in Velocity"),
    ylab = Some("Error in Time Lag"),
    plot_title = Some("Test Set Errors; Scatter")
  )

  write(
    tf_summary_dir/"test_errors.csv",
    "error_v,error_lag\n"+test_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))


  val test_scatter = plot_scatter(
    reg_metrics.preds,
    reg_time_lag.preds,
    xlab = Some("Velocity"),
    ylab = Some("Time Lag"),
    plot_title = Some("Training Set; Scatter"))

  hold()

  val test_actual_scatter = plot_scatter(
    reg_metrics.targets,
    reg_time_lag.targets)

  legend(Seq("Predictions", "Actual Data"))
  unhold()

  write(
    tf_summary_dir/"test_scatter.csv",
    "predv,predlag,actualv,actuallag\n"+
      test_scatter.zip(test_actual_scatter).map(
        c => c._1._1.toString + "," + c._1._2.toString + "," + c._2._1.toString + "," + c._2._2.toString
      ).mkString("\n")
  )

  val script = pwd/'helios/'scripts/"visualise_tl_results.R"

  try {
    %%('Rscript, script, tf_summary_dir)
  } catch {
    case e: Exception => e.printStackTrace()
  }

}


//Runs an experiment given some architecture, loss and training parameters.
def run_exp(
  dataset: TLDATA,
  architecture: Layer[Output, (Output, Output)],
  loss: Layer[((Output, Output), Output), Output],
  iterations: Int             = 150000,
  optimizer: Optimizer        = tf.train.AdaDelta(0.01),
  miniBatch: Int              = 512,
  sum_dir_prefix: String      = "",
  mo_flag: Boolean            = false,
  prob_timelags: Boolean      = false,
  timelag_pred_strategy: String = "mode"): ExperimentResult = {

  val train_fraction = 0.7

  val (data, collated_data): TLDATA = dataset

  val num_training = (collated_data.length*train_fraction).toInt
  val num_test = collated_data.length - num_training


  run_exp2(
    (
      (data.take(num_training), collated_data.take(num_training)),
      (data.takeRight(num_test), collated_data.takeRight(num_test))
    ),
    architecture, loss,
    iterations, optimizer, miniBatch, sum_dir_prefix,
    mo_flag, prob_timelags, timelag_pred_strategy
  )
}


def run_exp2(
  dataset: (TLDATA, TLDATA),
  architecture: Layer[Output, (Output, Output)],
  loss: Layer[((Output, Output), Output), Output],
  iterations: Int               = 150000,
  optimizer: Optimizer          = tf.train.AdaDelta(0.01),
  miniBatch: Int                = 512,
  sum_dir_prefix: String        = "",
  mo_flag: Boolean              = false,
  prob_timelags: Boolean        = false,
  timelag_pred_strategy: String = "mode"): ExperimentResult = {

  val (data, collated_data): TLDATA           = dataset._1
  val (data_test, collated_data_test): TLDATA = dataset._2

  val causal_window  = collated_data.head._2._2.length
  val num_training   = collated_data.length
  val num_test       = collated_data_test.length

  val model_train_eval = DataPipe(
    (dataTuple: ((HeliosDataSet, (GaussianScalerTF, GaussianScalerTF)), (Tensor, Tensor))) => {

      val ((tf_dataset, scalers), (train_time_lags, test_time_lags)) = dataTuple

      val training_data = tf.data.TensorSlicesDataset(tf_dataset.trainData)
        .zip(tf.data.TensorSlicesDataset(tf_dataset.trainLabels)).repeat()
        .shuffle(10)
        .batch(miniBatch)
        .prefetch(10)

      val dt = DateTime.now()

      val summary_dir_index  =
        if(mo_flag) sum_dir_prefix+"_timelag_inference_mo_"+dt.toString("YYYY-MM-dd-HH-mm")
        else sum_dir_prefix+"_timelag_inference_"+dt.toString("YYYY-MM-dd-HH-mm")

      val tf_summary_dir     = home/'tmp/summary_dir_index

      val input              = tf.learn.Input(FLOAT64, Shape(-1, tf_dataset.trainData.shape(1)))

      val trainInput         = tf.learn.Input(FLOAT64, Shape(-1, causal_window))

      val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

      val summariesDir       = java.nio.file.Paths.get(tf_summary_dir.toString())

      val (model, estimator) = dtflearn.build_tf_model(
        architecture, input, trainInput, trainingInputLayer,
        loss, optimizer, summariesDir,
        dtflearn.max_iter_stop(iterations))(
        training_data)

      val predictions: (Tensor, Tensor) = estimator.infer(() => tf_dataset.testData)


      val (pred_outputs_test, pred_time_lags_test) = process_predictions(
        predictions,
        causal_window,
        mo_flag,
        prob_timelags,
        timelag_pred_strategy,
        Some(scalers._2))

      val actual_outputs_test = (0 until num_test).map(n => {
        val time_lag = pred_time_lags_test(n).scalar.asInstanceOf[Double].toInt
        tf_dataset.testLabels(n, time_lag).scalar.asInstanceOf[Double]
      })

      val reg_metrics_time_lag = new RegressionMetricsTF(pred_time_lags_test, test_time_lags)
      val reg_metrics_output   = new RegressionMetricsTF(pred_outputs_test, actual_outputs_test)


      JointModelRun(
        (tf_dataset, scalers),
        model,
        estimator,
        (reg_metrics_output, reg_metrics_time_lag),
        tf_summary_dir,
        (train_time_lags, test_time_lags)
      )

    })

  //The processing pipeline
  val train_and_evaluate =
    data_splits_to_tensors(causal_window) >
      scale_data >
      model_train_eval

  val results_model_eval = train_and_evaluate(collated_data, collated_data_test)

  val exp_results = ExperimentResult(
    ExperimentType(mo_flag, prob_timelags, timelag_pred_strategy),
    (data, collated_data), (data_test, collated_data_test),
    results_model_eval
  )

  plot_and_write_results(exp_results)

  exp_results

}
