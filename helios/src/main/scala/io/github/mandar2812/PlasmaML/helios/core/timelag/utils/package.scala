package io.github.mandar2812.PlasmaML.helios.core.timelag

import ammonite.ops.{Path, write}
import breeze.linalg.{DenseMatrix, qr}
import breeze.stats.distributions.Gaussian
import _root_.io.github.mandar2812.dynaml.{DynaMLPipe => Pipe}
import io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.core.{CausalDynamicTimeLagSO, MOGrangerLoss, RBFWeightedSWLoss}
import io.github.mandar2812.PlasmaML.helios.data.HeliosDataSet
import io.github.mandar2812.PlasmaML.helios.fte
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.RandomVariable
import io.github.mandar2812.dynaml.tensorflow.data.TFDataSet
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.tensorflow.data._
import io.github.mandar2812.dynaml.tensorflow.utils.GaussianScalerTF
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer, Loss}
import org.platanios.tensorflow.api._

package object utils {

  //Define some types for convenience.

  type PATTERN     = ((Int, Tensor), (Float, Float))
  type SLIDINGPATT = (Int, (Tensor, Stream[Double], Float))

  type DATA        = Stream[PATTERN]
  type SLIDINGDATA = Stream[SLIDINGPATT]
  type TLDATA      = (DATA, SLIDINGDATA)

  type PROCDATA    = (HeliosDataSet, (Tensor, Tensor))

  type PROCDATA2   = (TFDataSet[(Tensor, Tensor)], (Tensor, Tensor))

  type NNPROP      = (Seq[Int], Seq[Shape], Seq[String], Seq[String])

  //Alias for the identity pipe/mapping
  def id[T]: DataPipe[T, T] = Pipe.identityPipe[T]

  //A Polynomial layer builder
  val layer_poly: Int => String => Activation = (power: Int) => (n: String) => new Activation(n) {
    override val layerType = "Poly"

    override protected def _forward(input: Output)(implicit mode: Mode): Output = {
      //val power = dtf.tensor_i32(input.shape(1))((1 to input.shape(1)):_*)

      input.pow(power)

    }
  }

  val getPolyAct: (Int, Int) => Int => Activation = (degree: Int, s: Int) => (i: Int) =>
    if(i - s == 0) layer_poly(degree)(s"Act_$i")
    else tf.learn.Sigmoid(s"Act_$i")

  val getReLUAct: Int => Int => Activation = (s: Int) => (i: Int) =>
    if((i - s) % 2 == 0) tf.learn.ReLU(s"Act_$i", 0.01f)
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


  /**
    * Subroutine to generate synthetic input-lagged output time series.
    *
    * x(n+1) = (1 - &alpha;). R(n) &times; x(n) + &epsilon;
    *
    * y(t + &Delta;t(x(t))) = f[x(t)]
    *
    * &Delta;t(x(t)) = g[x(t)]
    *
    * @param compute_output_and_lag A data pipe which takes the input x(t) a tensor and
    *                               computes y(t + &Delta;t(x(t))) the output and &Delta;(x(t)), the
    *                               causal time lag.
    *
    * @param d The dimensions in the input time series x(t)
    * @param n The length of the time series x(t)
    * @param noise The variance of &epsilon;
    * @param noiserot The variance of elements of R<sub>i,j</sub>,
    *                 a randomly generated matrix used to compute
    *                 an orthogonal transformation (rotation) of
    *                 x(t)
    * @param alpha A relaxation parameter which controls the
    *              auto-correlation time scale of x(t)
    *
    * @param sliding_window The size of the sliding time window [y(t), ..., y(t+h)]
    *                       to construct. This is used as training label for the model.
    *
    * @param confounding_factor A number between 0 and 1 determining how many input dimensions
    *                           should be retained in the generated data set. Defaults to 0, which
    *                           retains all the dimensions.
    * */
  def generate_data(
    compute_output_and_lag: DataPipe[Tensor, (Float, Float)],
    sliding_window: Int, d: Int = 3, n: Int = 5,
    noiserot: Double = 0.1, alpha: Double = 0.0,
    noise: Double = 0.5, confounding_factor: Double = 0d): TLDATA = {

    require(
      confounding_factor >= 0d && confounding_factor <= 1d,
      "The confounding factor can only be between 0 and 1")

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

    //Now slice the input tensors depending on the confounding.
    val num_sel_dims = d*math.ceil(1d - confounding_factor).toInt

    val data_con = data.map(patt => ((patt._1._1, patt._1._2(0 :: num_sel_dims)), patt._2))

    val joined_data_con = joined_data.map(patt => (patt._1, (patt._2._1(0 :: num_sel_dims), patt._2._2, patt._2._3)))

    (data_con, joined_data_con)
  }

  /**
    * Plot the synthetic data set produced by [[generate_data()]].
    * */
  def plot_data(dataset: TLDATA): Unit = {

    val (data, joined_data) = dataset

    val sliding_window = joined_data.head._2._2.length

    val (causes, effects) = data.unzip

    val energies = data.map(_._2._2)

    /*spline(energies)
    title("Output Time Series")*/

    val effect_times = data.map(_._2._1)

    val outputs = effects.groupBy(_._1.toInt).mapValues(v => v.map(_._2).sum/v.length.toDouble).toSeq.sortBy(_._1)

    try {

      histogram(effect_times.zip(causes.map(_._1)).map(c => c._1 - c._2))
      title("Distribution of time lags")

    } catch {
      case _: java.util.NoSuchElementException => println("Can't plot histogram due to `No Such Element` exception")
      case _: Throwable => println("Can't plot histogram due to exception")
    }

    line(outputs)
    hold()
    line(energies)
    legend(Seq("Output Data with Lag", "Output Data without Lag"))
    unhold()

    spline(autocorrelation(2*sliding_window)(data.map(_._2._2.toDouble)))
    title("Auto-covariance of time series")

  }

  /**
    * Creates a data pipeline which takes a [[SLIDINGDATA]] object
    * and returns a [[PROCDATA]] instance.
    *
    * The pipeline splits the data set into training and test then loads
    * them into a [[HeliosDataSet]] object. Uses the [[data_splits_to_tensors()]]
    * method.
    *
    * The ground truth time lags are also returned.
    *
    * @param num_training The size of the training data set.
    * @param num_test The size of the test data set.
    * @param sliding_window The size of the causal time window.
    *
    * */
  def load_data_into_tensors(num_training: Int, num_test: Int, sliding_window: Int)
  : DataPipe[SLIDINGDATA, PROCDATA] = DataPipe((data: SLIDINGDATA) => {

    require(
      num_training + num_test == data.length,
      "Size of train and test data "+ "must add up to total size of data!"
    )

    (data.take(num_training), data.takeRight(num_test))}) >
    data_splits_to_tensors(sliding_window)

  /**
    * Takes training and test data sets of type [[SLIDINGDATA]]
    * and loads them in an object of type [[PROCDATA]].
    *
    * @param sliding_window The size of the causal time window.
    *
    * */
  def data_splits_to_tensors(sliding_window: Int): DataPipe2[SLIDINGDATA, SLIDINGDATA, PROCDATA] =
    DataPipe2((training_data: SLIDINGDATA, test_data: SLIDINGDATA) => {

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


  /**
    * Takes training and test data sets of type [[SLIDINGDATA]]
    * and loads them in an object of type [[PROCDATA2]].
    *
    * [[PROCDATA2]] consists of a DynaML [[TFDataSet]] object,
    * along with the ground truth causal time lags for the
    * train and test sets.
    *
    * @param causal_window The size of the causal time window.
    *
    * */
  def data_splits_to_dataset(causal_window: Int): DataPipe2[SLIDINGDATA, SLIDINGDATA, PROCDATA2] =
    DataPipe2(
      (training_data: SLIDINGDATA, test_data: SLIDINGDATA) => {

        //Get the ground truth values of the causal time lags.
        val (train_time_lags, test_time_lags): (Tensor, Tensor) = (
          dtf.tensor_f64(training_data.length)(training_data.toList.map(d => d._2._3.toDouble):_*),
          dtf.tensor_f64(test_data.length)(test_data.toList.map(d => d._2._3.toDouble):_*))

        //Create the data set
        val train_dataset = dtfdata.dataset(training_data).map(
          (p: SLIDINGPATT) => (p._2._1, dtf.tensor_f64(causal_window)(p._2._2:_*))
        )

        val test_dataset = dtfdata.dataset(test_data).map(
          (p: SLIDINGPATT) => (p._2._1, dtf.tensor_f64(causal_window)(p._2._2:_*))
        )

        val tf_dataset = TFDataSet(train_dataset, test_dataset)

        (tf_dataset, (train_time_lags, test_time_lags))
      })


  /**
    * Scale training features/labels, on the test data; apply scaling
    * only to the features.
    *
    * */
  val scale_helios_dataset = DataPipe((dataset: HeliosDataSet) => {

    val (norm_tr_data, scalers) = dtfpipe.gaussian_standardization(dataset.trainData, dataset.trainLabels)

    (
      dataset.copy(
        trainData = norm_tr_data._1, trainLabels = norm_tr_data._2,
        testData = scalers._1(dataset.testData)),
      scalers
    )
  })

  /**
    * Data pipeline used by [[run_exp_joint()]] and [[run_exp_stage_wise()]]
    * methods to scale data before training.
    * */
  val scale_data_v1 = DataPipe(
    scale_helios_dataset,
    id[(Tensor, Tensor)]
  )

  /**
    * Data pipeline used by [[run_exp_hyp()]]
    * method to scale data before training.
    * */
  val scale_data_v2 = DataPipe(
    fte.scale_dataset, id[(Tensor, Tensor)]
  )

  /**
    * Returns the properties [[NNPROP]] (i.e. layer sizes, shapes, parameter names, & data types)
    * of a feed-forward/dense neural stack which consists of layers of equal size.
    *
    * @param d The dimensionality of the input (assumed to be a rank 1 tensor).
    * @param num_pred_dims The dimensionality of the network output.
    * @param num_neurons The size of each hidden layer.
    * @param num_hidden_layers The number of hidden layers.
    *
    * */
  def get_ffnet_properties(
    d: Int, num_pred_dims: Int,
    num_neurons: Int,
    num_hidden_layers: Int): NNPROP = {

    val net_layer_sizes       = Seq(d) ++ Seq.fill(num_hidden_layers)(num_neurons) ++ Seq(num_pred_dims)
    val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))
    val layer_parameter_names = (1 to net_layer_sizes.tail.length).map(s => "Linear_"+s+"/Weights")
    val layer_datatypes       = Seq.fill(net_layer_sizes.tail.length)("FLOAT64")

    (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes)
  }

  /**
    * Returns the properties [[NNPROP]] (i.e. layer sizes, shapes, parameter names, & data types)
    * of a feed-forward/dense neural stack which consists of layers of unequal size.
    *
    * @param d The dimensionality of the input (assumed to be a rank 1 tensor).
    * @param num_pred_dims The dimensionality of the network output.
    * @param layer_sizes The size of each hidden layer.
    * @param dType The data type of the layer weights and biases.
    * @param starting_index The numeric index of the first layer, defaults to 1.
    *
    * */
  def get_ffnet_properties(
    d: Int, num_pred_dims: Int,
    layer_sizes: Seq[Int],
    dType: String = "FLOAT64",
    starting_index: Int = 1): NNPROP = {

    val net_layer_sizes       = Seq(d) ++ layer_sizes ++ Seq(num_pred_dims)

    val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))

    val size                  = net_layer_sizes.tail.length

    val layer_parameter_names = (starting_index until starting_index + size).map(i => s"Linear_$i/Weights")

    val layer_datatypes       = Seq.fill(net_layer_sizes.tail.length)(dType)

    (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes)
  }

  /**
    * Creates an output mapping layer which
    * produces outputs in the form desired by
    * time lag based loss functions in [[helios.core]].
    *
    * @param causal_window The size of the sliding causal time window.
    * @param mo_flag Set to true if the model produces predictions for each
    *                time step in the causal window.
    * @param prob_timelags Set to true if the time lag prediction is in the
    *                      form of a probability distribution over the causal
    *                      time window.
    * @param time_scale An optional parameter, used only if `mo_flag` and
    *                   `prob_timelags` are both set to false.
    * */
  def get_output_mapping(
    causal_window: Int,
    mo_flag: Boolean,
    prob_timelags: Boolean,
    dist_type: String,
    time_scale: Double = 1.0): Layer[Output, (Output, Output)] = if (!mo_flag) {

    if (!prob_timelags) RBFWeightedSWLoss.output_mapping("Output/RBFWeightedL1", causal_window, time_scale)
    else CausalDynamicTimeLagSO.output_mapping("Output/SOProbWeightedTS", causal_window)

  } else if(mo_flag && !prob_timelags) {

    MOGrangerLoss.output_mapping("Output/MOGranger", causal_window)

  } else {
    dist_type match {
      case "poisson"  => helios.learn.cdt_poisson_loss.output_mapping(
        name = "Output/PoissonWeightedTS",
        causal_window)

      case "beta"     => helios.learn.cdt_beta_loss.output_mapping(
        name = "Output/BetaWeightedTS",
        causal_window)

      case "gaussian" => helios.learn.cdt_gaussian_loss.output_mapping(
        name = "Output/GaussianWeightedTS",
        causal_window)

      case _          => helios.learn.cdt_loss.output_mapping(
        name = "Output/ProbWeightedTS",
        causal_window)
    }
  }

  /**
    * Calculate the size of the
    * penultimate layer of a neural stack
    * used for causal time lag prediction.
    * @param causal_window The size of the sliding causal time window.
    * @param mo_flag Set to true if the model produces predictions for each
    *                time step in the causal window.
    * @param prob_timelags Set to true if the time lag prediction is in the
    *                      form of a probability distribution over the causal
    *                      time window.
    * */
  def get_num_output_dims(
    causal_window: Int,
    mo_flag: Boolean,
    prob_timelags: Boolean,
    dist_type: String): Int =
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


  def get_train_hooks(
    p: Path, it: Int,
    epochFlag: Boolean,
    num_data: Int,
    batch_size: Int) = if(epochFlag) {

    val epochSize = num_data/batch_size
    dtflearn.model._train_hooks(p, it*epochSize/3, it*epochSize/3, it*epochSize)
  } else {
    dtflearn.model._train_hooks(p, it/3, it/3, it)
  }

  def get_stop_condition(
    it: Int,
    tol: Double,
    epochF: Boolean,
    num_data: Int,
    batch_size: Int): StopCriteria = if(epochF) {

    val epochSize = num_data/batch_size

    tf.learn.StopCriteria(
      maxSteps = Some(it*epochSize),
      maxEpochs = Some(it),
      relLossChangeTol = Some(tol))
  } else {
    dtflearn.rel_loss_change_stop(tol, it)
  }

  /**
    * Get the appropriate causal time lag loss function.
    * @param sliding_window The size of the sliding causal time window.
    * @param mo_flag Set to true if the model produces predictions for each
    *                time step in the causal window.
    * @param prob_timelags Set to true if the time lag prediction is in the
    *                      form of a probability distribution over the causal
    *                      time window.
    * */
  def get_loss(
    sliding_window: Int,
    mo_flag: Boolean,
    prob_timelags: Boolean,
    p: Double                     = 1.0,
    time_scale: Double            = 1.0,
    corr_sc: Double               = 2.5,
    c_cutoff: Double              = 0.0,
    prior_wt: Double              = 1d,
    prior_divergence:  helios.learn.cdt_loss.Divergence = helios.learn.cdt_loss.KullbackLeibler,
    temp: Double                  = 1.0,
    error_wt: Double              = 1.0,
    c: Double                     = 1.0): Loss[((Output, Output), Output)] =
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
          error_wt = error_wt,
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

  def collect_predictions(preds: DataSet[(Tensor, Tensor)]): (Tensor, Tensor) =
    (
      tfi.stack(preds.map((p: (Tensor, Tensor)) => p._1).data.toSeq, axis = -1),
      tfi.stack(preds.map((p: (Tensor, Tensor)) => p._2).data.toSeq, axis = -1)
    )

  /**
    * Process the predictions made by a causal time lag model.
    *
    * */
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

  /**
    * Takes a tensor of rank 1 (Shape(n)) and plots a histogram.
    * */
  @throws[java.util.NoSuchElementException]
  @throws[Exception]
  def plot_histogram(data: Tensor, plot_title: String): Unit = {
    try {

      histogram(dtfutils.toDoubleSeq(data).toSeq)
      title(plot_title)

    } catch {
      case _: java.util.NoSuchElementException => println("Can't plot histogram due to `No Such Element` exception")
      case _: Throwable => println("Can't plot histogram due to exception")
    }
  }

  /**
    * Plot input-output pairs as a scatter plot.
    *
    * @param input A Stream of input patterns.
    * @param input_to_scalar A function which processes each multi-dimensional
    *                        pattern to a scalar value.
    * @param targets A tensor containing the ground truth values of the target
    *                function.
    * @param predictions A tensor containing the model predictions.
    * @param xlab x-axis label
    * @param ylab y-axis label
    * @param plot_title The plot title, as a string.
    * */
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

  /**
    * Plot multiple input-output pairs on a scatter plot.
    *
    * @param input A Stream of input patterns.
    * @param input_to_scalar A function which processes each multi-dimensional
    *                        pattern to a scalar value.
    * @param predictions A sequence of tensors containing the predictions for each model/predictor.
    * @param xlab x-axis label
    * @param ylab y-axis label
    * @param plot_legend A sequence of labels for each model/predictor,
    *                    to be displayed as the plot legend.
    * @param plot_title The plot title, as a string.
    * */
  def plot_input_output(
    input: Stream[Tensor],
    input_to_scalar: Tensor => Double,
    predictions: Seq[Tensor],
    xlab: String,
    ylab: String,
    plot_legend: Seq[String],
    plot_title: String): Unit = {

    val processed_inputs = input.map(input_to_scalar)

    scatter(processed_inputs.zip(dtfutils.toDoubleSeq(predictions.head).toSeq))
    hold()
    predictions.tail.foreach(pred => {
      scatter(processed_inputs.zip(dtfutils.toDoubleSeq(pred).toSeq))
    })

    xAxis(xlab)
    yAxis(ylab)
    title(plot_title)
    legend(plot_legend)
    unhold()
  }

  /**
    * Plot input-output pairs as a scatter plot.
    *
    * @param xy The x and y axis data, a sequence of tuples
    * @param xlab x-axis label
    * @param ylab y-axis label
    * @param plot_title The plot title, as a string.
    * */
  def plot_scatter(
    xy: Seq[(Double, Double)],
    xlab: Option[String] = None,
    ylab: Option[String] = None,
    plot_title: Option[String] = None): Unit = {
    scatter(xy)
    if(xlab.isDefined) xAxis(xlab.get)
    if(ylab.isDefined) yAxis(ylab.get)
    if(plot_title.isDefined) title(plot_title.get)
  }

  def write_data_set(
    data: TLDATA,
    summary_dir: Path,
    identifier: String): Unit = {

    //Write the features.
    write(
      summary_dir/s"${identifier}_features.csv",
      data._2.map(_._2._1)
        .map(x => dtfutils.toDoubleSeq(x).mkString(","))
        .mkString("\n")
    )

    write(
      summary_dir/s"${identifier}_output_lag.csv",
      data._1.map(_._2)
        .map(x => s"${x._1},${x._2}")
        .mkString("\n"))

    //Write the slided outputs.
    write(
      summary_dir/s"${identifier}_targets.csv",
      data._2.map(_._2._2)
        .map(_.mkString(","))
        .mkString("\n")
    )

  }

  def write_model_outputs(
    outputs: (Tensor, Tensor),
    summary_dir: Path,
    identifier: String): Unit = {

    val h = outputs._1.shape(1)

    write(
      summary_dir/s"${identifier}_predictions.csv",
      dtfutils.toDoubleSeq(outputs._1).grouped(h).map(_.mkString(",")).mkString("\n")
    )

    write(
      summary_dir/s"${identifier}_probabilities.csv",
      dtfutils.toDoubleSeq(outputs._2).grouped(h).map(_.mkString(",")).mkString("\n")
    )

  }

  def write_predictions_and_gt(
    predictions: Seq[(Double, Double)],
    ground_truth: Seq[(Double, Double)],
    summary_dir: Path, identifier: String): Unit = {

    write(
      summary_dir/s"${identifier}_scatter.csv",
      "predv,predlag,actualv,actuallag\n"+
        predictions.zip(ground_truth).map(
          c => s"${c._1._1},${c._1._2},${c._2._1},${c._2._2}"
        ).mkString("\n")
    )
  }

}
