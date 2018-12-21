package io.github.mandar2812.PlasmaML

import ammonite.ops._
import org.joda.time._
import com.sksamuel.scrimage.Image
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.evaluation.{ClassificationMetricsTF, RegressionMetricsTF}
import io.github.mandar2812.dynaml.tensorflow.{dtf, dtflearn, dtfutils}
import io.github.mandar2812.dynaml.tensorflow.data.{TFDataSet, DataSet}
import io.github.mandar2812.dynaml.tensorflow.implicits._
import io.github.mandar2812.PlasmaML.helios.core._
import io.github.mandar2812.PlasmaML.helios.data._
import io.github.mandar2812.PlasmaML.dynamics.mhd._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria, SupervisedTrainableModel}
import org.platanios.tensorflow.api.learn.layers.{Compose, Layer, Loss}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.types.DataType
import spire.math.UByte
import _root_.io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import org.platanios.tensorflow.api.learn.estimators.Estimator

/**
  * <h3>Helios</h3>
  *
  * The top level package for the helios module.
  *
  * Contains methods for carrying out ML experiments
  * on data sets associated with helios.
  *
  * @author mandar2812
  * */
package object helios {

  /**
    * Calculate RMSE of a tensorflow based estimator.
    * */
  def calculate_rmse(
    n: Int, n_part: Int)(
    labels_mean: Tensor, labels_stddev: Tensor)(
    images: Tensor, labels: Tensor)(infer: Tensor => Tensor): Float = {

    def accuracy(im: Tensor, lab: Tensor): Float = {
      infer(im)
        .multiply(labels_stddev)
        .add(labels_mean)
        .subtract(lab).cast(FLOAT64)
        .square.mean().scalar
        .asInstanceOf[Float]
    }

    val num_elem: Int = n/n_part

    math.sqrt((0 until n_part).map(i => {

      val (lower_index, upper_index) = (i*num_elem, if(i == n_part-1) n else (i+1)*num_elem)

      accuracy(images(lower_index::upper_index, ::, ::, ::), labels(lower_index::upper_index))
    }).sum/num_elem).toFloat
  }

  object learn {

    val upwind_1d: UpwindTF.type                              = UpwindTF

    /*
    * NN Architectures
    *
    * */
    val cnn_goes_v1: Layer[Output, Output]                    = Arch.cnn_goes_v1
    val cnn_goes_v1_1: Layer[Output, Output]                  = Arch.cnn_goes_v1_1
    val cnn_goes_v1_2: Layer[Output, Output]                  = Arch.cnn_goes_v1_2
    val cnn_sw_v1: Layer[Output, Output]                      = Arch.cnn_sw_v1
    val cnn_sw_dynamic_timescales_v1: Layer[Output, Output]   = Arch.cnn_sw_dynamic_timescales_v1
    val cnn_xray_class_v1: Layer[Output, Output]              = Arch.cnn_xray_class_v1

    def cnn_sw_v2(sliding_window: Int): Compose[Output, Output, Output] =
      Arch.cnn_sw_v2(sliding_window, mo_flag = true, prob_timelags = true)

    /*
    * Loss Functions
    * */
    val weightedL2FluxLoss: WeightedL2FluxLoss.type            = WeightedL2FluxLoss
    val rBFWeightedSWLoss: RBFWeightedSWLoss.type              = RBFWeightedSWLoss
    val dynamicRBFSWLoss: DynamicRBFSWLoss.type                = DynamicRBFSWLoss
    val cdt_loss: CausalDynamicTimeLag.type                    = CausalDynamicTimeLag
    val cdt_i: CausalDynamicTimeLagI.type                      = CausalDynamicTimeLagI
    val cdt_ii: CausalDynamicTimeLagII.type                    = CausalDynamicTimeLagII
    val cdt_loss_so: CausalDynamicTimeLagSO.type               = CausalDynamicTimeLagSO
    val cdt_poisson_loss: WeightedTimeSeriesLossPoisson.type   = WeightedTimeSeriesLossPoisson
    val cdt_gaussian_loss: WeightedTimeSeriesLossGaussian.type = WeightedTimeSeriesLossGaussian
    val cdt_beta_loss: WeightedTimeSeriesLossGaussian.type     = WeightedTimeSeriesLossGaussian
  }

  /**
    * A model run contains a tensorflow model/estimator as
    * well as its training/test data set and meta data regarding
    * the training/evaluation process.
    *
    * */
  sealed trait ModelRun {

    type DATA_PATTERN
    type SCALERS
    type MODEL
    type ESTIMATOR

    val summary_dir: Path

    val data_and_scales: (TFDataSet[DATA_PATTERN], SCALERS)

    val metrics_train: Option[RegressionMetricsTF]

    val metrics_test: Option[RegressionMetricsTF]

    val model: MODEL

    val estimator: ESTIMATOR

  }

  case class SupervisedModelRun[X, T, ModelOutput, ModelOutputSym](
    data_and_scales: (TFDataSet[(X, T)], (ReversibleScaler[X], ReversibleScaler[T])),
    model: SupervisedTrainableModel[
      Tensor, Output, DataType, Shape, ModelOutputSym,
      Tensor, Output, DataType, Shape, Output],
    estimator: Estimator[
      Tensor, Output, DataType, Shape, ModelOutputSym,
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
      (ModelOutputSym, Output)],
    metrics_train: Option[RegressionMetricsTF],
    metrics_test: Option[RegressionMetricsTF],
    summary_dir: Path,
    training_preds: Option[ModelOutput],
    test_preds: Option[ModelOutput]) extends ModelRun {

    override type DATA_PATTERN = (X, T)

    override type SCALERS = (ReversibleScaler[X], ReversibleScaler[T])

    override type MODEL = SupervisedTrainableModel[
      Tensor, Output, DataType, Shape, ModelOutputSym,
      Tensor, Output, DataType, Shape, Output]

    override type ESTIMATOR = Estimator[
      Tensor, Output, DataType, Shape, ModelOutputSym,
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
      (ModelOutputSym, Output)]
  }

  case class ExperimentType(
    multi_output: Boolean,
    probabilistic_time_lags: Boolean,
    timelag_prediction: String)

  case class ExperimentResult[DATA, X, Y, ModelOutput, ModelOutputSym](
    config: ExperimentType,
    train_data: DATA,
    test_data: DATA,
    results: SupervisedModelRun[X, Y, ModelOutput, ModelOutputSym])


  /**
    * Train a Neural architecture on a
    * processed data set.
    *
    * @param collated_data Data set of temporally joined
    *                      image paths and GOES X-Ray fluxes.
    *                      This is generally the output after
    *                      executing [[collate_goes_data_range()]]
    *                      with the relevant parameters.
    * @param tt_partition A function which splits the data set
    *                     into train and test sections, based on
    *                     any Boolean function. If the function
    *                     returns true then the instance falls into
    *                     the training set else the test set
    * @param resample If set to true, the training data is resampled
    *                 to balance the occurrence of high flux and low
    *                 flux events.
    * @param longWavelength If set to true, predict long wavelength
    *                       GOES X-Ray flux, else short wavelength,
    *                       defaults to short wavelength.
    * @param tempdir A working directory where the results will be
    *                archived, defaults to user_home_dir/tmp. The model
    *                checkpoints and other results will be stored inside
    *                another directory created in tempdir.
    * @param results_id The suffix added the results/checkpoints directory name.
    * @param max_iterations The maximum number of iterations that the
    *                       network must be trained for.
    * @param arch The neural architecture to train, defaults to [[Arch.cnn_goes_v1]]
    * @param lossFunc The loss function which will be used to guide the training
    *                 of the architecture, defaults to [[tf.learn.L2Loss]]
    *
    * */
  def run_experiment_goes(
    collated_data: Stream[(DateTime, (Path, (Double, Double)))],
    tt_partition: ((DateTime, (Path, (Double, Double)))) => Boolean,
    resample: Boolean = false, longWavelength: Boolean = false)(
    results_id: String, max_iterations: Int,
    tempdir: Path = home/"tmp",
    arch: Layer[Output, Output] = learn.cnn_goes_v1,
    lossFunc: Loss[(Output, Output)] = tf.learn.L2Loss("Loss/L2")) = {

    val resDirName = if(longWavelength) "helios_goes_long_"+results_id else "helios_goes_"+results_id

    val tf_summary_dir = tempdir/resDirName


    val checkpoints =
      if (exists(tf_summary_dir)) ls! tf_summary_dir |? (_.isFile) |? (_.segments.last.contains("model.ckpt-"))
      else Seq()

    val checkpoint_max =
      if(checkpoints.isEmpty) 0
      else (checkpoints | (_.segments.last.split("-").last.split('.').head.toInt)).max

    val iterations = if(max_iterations > checkpoint_max) max_iterations - checkpoint_max else 0


    /*
    * After data has been joined/collated,
    * start loading it into tensors
    *
    * */

    val dataSet = helios.data.prepare_helios_goes_data_set(
      collated_data,
      tt_partition,
      scaleDownFactor = 2,
      resample)

    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainData)

    val targetIndex = if(longWavelength) 1 else 0

    val train_labels = dataSet.trainLabels(::, targetIndex)

    val labels_mean = train_labels.mean()

    val labels_stddev = train_labels.subtract(labels_mean).square.mean().sqrt

    val norm_train_labels = train_labels.subtract(labels_mean).divide(labels_stddev)

    val trainLabels = tf.data.TensorSlicesDataset(norm_train_labels)

    //val trainWeights = tf.data.TensorSlicesDataset(norm_train_labels.exp)

    val trainData =
      trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(64)
        .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")
    val input = tf.learn.Input(
      UINT8,
      Shape(
        -1,
        dataSet.trainData.shape(1),
        dataSet.trainData.shape(2),
        dataSet.trainData.shape(3))
    )

    val trainInput = tf.learn.Input(FLOAT64, Shape(-1))

    val trainingInputLayer = tf.learn.Cast("TrainInput", INT64)

    val loss = lossFunc >>
      tf.learn.Mean("Loss/Mean") >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    val optimizer = tf.train.AdaGrad(0.002)

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    //Now create the model
    val (model, estimator) = tf.createWith(graph = Graph()) {
      val model = tf.learn.Model.supervised(
        input, arch, trainInput, trainingInputLayer,
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


    //Create  MetricsTF instance
    //First calculate and re-normalize the test predictions

    val predictions = estimator.infer(() => dataSet.testData)
      .multiply(labels_stddev)
      .add(labels_mean)
      .reshape(Shape(dataSet.nTest))

    val targets = dataSet.testLabels(::, targetIndex)
    val metrics = new RegressionMetricsTF(predictions, targets)

    val (predictions_seq, targets_seq) = (
      predictions.entriesIterator.map(_.asInstanceOf[Float]).map(GOESData.getFlareClass).toSeq,
      targets.entriesIterator.map(_.asInstanceOf[Float]).map(GOESData.getFlareClass).toSeq)

    val preds_one_hot = dtf.tensor_i32(dataSet.nTest)(predictions_seq:_*).oneHot(depth = 4)
    val targets_one_hot = dtf.tensor_i32(dataSet.nTest)(targets_seq:_*).oneHot(depth = 4)

    val metrics_class = new ClassificationMetricsTF(4, preds_one_hot, targets_one_hot)

    (model, estimator, metrics, metrics_class, tf_summary_dir, labels_mean, labels_stddev, collated_data)
  }

  def write_predictions(
    preds: Seq[Double],
    targets: Seq[Double],
    timelags: Seq[Double],
    file: Path): Unit = {

    //write predictions and ground truth to a csv file

    val resScatter = preds.zip(targets).zip(timelags).map(p => Seq(p._1._1, p._1._2, p._2))

    streamToFile(file.toString())(resScatter.map(_.mkString(",")).toStream)
  }

  def visualise_cdt_results(results_path: Path): Unit = {

    val split_line = DataPipe[String, Array[Double]](_.split(',').map(_.toDouble))

    val process_pipe = DataPipe[Path, String](_.toString()) >
      fileToStream >
      StreamDataPipe(split_line)

    val scatter_data = process_pipe(results_path)

    regression(scatter_data.map(p => (p.head, p(1))))
    hold()
    title("Scatter Plot: Model Predictions")
    xAxis("Prediction")
    yAxis("Actual Target")
    unhold()

    scatter(scatter_data.map(p => (p.head, p.last)))
    hold()
    title("Scatter Plot: Output Timelag Relationship")
    xAxis("Prediction")
    yAxis("Predicted Time Lag")
    unhold()

  }

  def run_unsupervised_experiment(
    collated_data: HELIOS_IMAGE_DATA,
    tt_partition: IMAGE_PATTERN => Boolean,
    resample: Boolean = false,
    preprocess_image: DataPipe[Image, Image] = identityPipe[Image],
    image_to_bytearr: DataPipe[Image, Array[Byte]] = DataPipe((i: Image) => i.argb.flatten.map(_.toByte)),
    processed_image_size: (Int, Int) = (-1, -1),
    num_channels_image: Int = 4,
    image_history: Int = 0,
    image_history_downsampling: Int = 1)(
    results_id: String,
    stop_criteria: StopCriteria,
    tempdir: Path = home/"tmp",
    arch: Layer[Output, (Output, Output)],
    lossFunc: Layer[(Output, (Output, Output)), Output],
    optimizer: Optimizer = tf.train.AdaDelta(0.001),
    miniBatchSize: Int = 16,
    reuseExistingModel: Boolean = false) = {

    //The directories to write model parameters and summaries.
    val resDirName = if(reuseExistingModel) results_id else "helios_"+results_id

    val tf_summary_dir = tempdir/resDirName

    val load_image_into_tensor = data.read_image >
      preprocess_image >
      image_to_bytearr >
      DataPipe(
        dtf.tensor_from_buffer(
          dtype = "UINT8", processed_image_size._1,
          processed_image_size._2,
          num_channels_image) _
      )


    val dataSet: TF_IMAGE_DATA = helios.data.prepare_helios_data_set(
      collated_data,
      load_image_into_tensor,
      tt_partition,
      image_history,
      image_history_downsampling)

    val data_shape = Shape(
      processed_image_size._1,
      processed_image_size._2,
      num_channels_image*(image_history_downsampling + 1)
    )

    val train_data = dataSet.training_dataset.build[Tensor, Output, DataType.Aux[UByte], DataType, Shape](
      Left(identityPipe[Tensor]),
      dataType = UINT8, data_shape)
      .repeat()
      .shuffle(1000)
      .batch(miniBatchSize)
      .prefetch(10)


    /*
    * Start building tensorflow network/graph
    * */
    println("Building the unsupervised model.")
    val input = tf.learn.Input(
      UINT8, Shape(-1) ++ data_shape
    )


    val split_stacked_images = new Layer[Output, Seq[Output]]("UnstackImages") {
      override val layerType: String = "UnstackImage"

      override protected def _forward(input: Output)(implicit mode: Mode): Seq[Output] = {

        tf.splitEvenly(input, image_history_downsampling + 1, -1)
      }
    }

    val write_images = (image_name: String) =>
      dtflearn.seq_layer(
        "WriteImagesTS",
        Seq.tabulate(image_history_downsampling + 1)(
          i => tf.learn.ImageSummary(s"ImageSummary_$i", s"${image_name}_$i", maxOutputs = miniBatchSize/2))
      ) >> dtflearn.concat_outputs("Concat_Images")



    val loss = dtflearn.tuple2_layer(
      "Tup2",
      split_stacked_images >> write_images("Actual_Image"),
      dtflearn.tuple2_layer(
        "Tup_1",
        dtflearn.identity[Output]("Id"),
        split_stacked_images >> write_images("Reconstruction"))) >>
      lossFunc >>
      tf.learn.ScalarSummary("Loss", "ReconstructionLoss")

    //Now train the model
    val (model, estimator) = dtflearn.build_unsup_tf_model(
      arch, input, loss,
      optimizer, java.nio.file.Paths.get(tf_summary_dir.toString()),
      stop_criteria, stepRateFreq = 5000, summarySaveFreq = 5000, checkPointFreq = 5000)(
      train_data, inMemory = false)


    (model, estimator, collated_data, dataSet, tf_summary_dir)

  }

  /**
    * Train and test a CNN based solar wind prediction architecture.
    *
    * @param collated_data Data set of temporally joined
    *                      image paths and GOES X-Ray fluxes.
    *                      This is generally the output after
    *                      executing [[collate_goes_data_range()]]
    *                      with the relevant parameters.
    * @param tt_partition A function which splits the data set
    *                     into train and test sections, based on
    *                     any Boolean function. If the function
    *                     returns true then the instance falls into
    *                     the training set else the test set
    * @param resample If set to true, the training data is resampled
    *                 to balance the occurrence of high flux and low
    *                 flux events.
    *
    * @param tempdir A working directory where the results will be
    *                archived, defaults to user_home_dir/tmp. The model
    *                checkpoints and other results will be stored inside
    *                another directory created in tempdir.
    * @param results_id The suffix added the results/checkpoints directory name.
    * @param stop_criteria When to stop training, an instance of [[StopCriteria]]
    * @param arch The neural architecture to train, for example see [[learn.cnn_sw_v1]]
    *
    * */
  def run_cdt_experiment_omni(
    collated_data: HELIOS_OMNI_DATA,
    tt_partition: PATTERN => Boolean,
    resample: Boolean = false,
    preprocess_image: DataPipe[Image, Image] = identityPipe[Image],
    image_to_bytearr: DataPipe[Image, Array[Byte]] = DataPipe((i: Image) => i.argb.flatten.map(_.toByte)),
    processed_image_size: (Int, Int) = (-1, -1),
    num_channels_image: Int = 4,
    image_history: Int = 0,
    image_history_downsampling: Int = 1)(
    results_id: String,
    stop_criteria: StopCriteria,
    tempdir: Path = home/"tmp",
    arch: Layer[Output, (Output, Output)],
    lossFunc: Layer[((Output, Output), Output), Output],
    mo_flag: Boolean = true,
    prob_timelags: Boolean = true,
    optimizer: Optimizer = tf.train.AdaDelta(0.001),
    miniBatchSize: Int = 16,
    reuseExistingModel: Boolean = false): ExperimentResult[DataSet[PATTERN], Tensor, Tensor, (Tensor, Tensor), (Output, Output)] = {

    //The directories to write model parameters and summaries.
    val resDirName = if(reuseExistingModel) results_id else "helios_omni_"+results_id

    val tf_summary_dir = tempdir/resDirName

    val num_outputs = collated_data.data.head._2._2.length

    /*
    * Create the data processing pipe for processing each image.
    *
    * 1. Load the image into a scrimage object
    * 2. Apply pre-processing filters on the image
    * 3. Convert the processed image into a byte array
    * 4. Load the byte array into a tensor
    * */


    val image_into_tensor = data.read_image >
      preprocess_image >
      image_to_bytearr >
      DataPipe((arr: Array[Byte]) => dtf.tensor_from_buffer(
        dtype = "UINT8", processed_image_size._1,
        processed_image_size._2,
        num_channels_image)(arr))


    val load_image: DataPipe[Seq[Path], Option[Tensor]] = DataPipe((images: Seq[Path]) => {

      val first_non_corrupted_file: (Int, Path) =
        images
          .map(p => (available_bytes(p), p))
          .sortBy(_._1)
          .reverse
          .head

      if (first_non_corrupted_file._1 > 0) Some(image_into_tensor(first_non_corrupted_file._2)) else None

    })

    val load_targets_into_tensor = DataPipe((arr: Seq[Double]) => dtf.tensor_f32(num_outputs)(arr:_*))

    val dataSet: TF_DATA = helios.data.prepare_helios_data_set(
      collated_data,
      load_image,
      load_targets_into_tensor,
      tt_partition,
      resample,
      image_history,
      image_history_downsampling)

    val (norm_tf_data, scalers): SC_TF_DATA = scale_helios_dataset(dataSet)

    val causal_horizon = collated_data.data.head._2._2.length

    val data_shapes = (
      Shape(processed_image_size._1, processed_image_size._2, num_channels_image*(image_history_downsampling + 1)),
      Shape(causal_horizon)
    )

    val trainData = norm_tf_data.training_dataset.build[
        (Tensor, Tensor), (Output, Output),
        (DataType.Aux[UByte], DataType.Aux[Float]), (DataType, DataType),
        (Shape, Shape)](
      Left(identityPipe[(Tensor, Tensor)]),
      dataType = (UINT8, FLOAT32), data_shapes)
      .repeat()
      .shuffle(1000)
      .batch(miniBatchSize)
      .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")
    val input = tf.learn.Input(
      UINT8, Shape(-1) ++ data_shapes._1
    )

    val trainInput = tf.learn.Input(FLOAT32, Shape(-1) ++ data_shapes._2)

    val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

    val loss = lossFunc >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    //Now create the model
    val (model, estimator) = dtflearn.build_tf_model(
      arch, input, trainInput, trainingInputLayer,
      loss, optimizer, java.nio.file.Paths.get(tf_summary_dir.toString()),
      stop_criteria)(
      trainData)

    val nTest = norm_tf_data.test_dataset.size

    val predictions: (Tensor, Tensor) = dtfutils.buffered_preds[
      Tensor, Output, DataType, Shape, (Output, Output),
      Tensor, Output, DataType, Shape, Output,
      Tensor, (Tensor, Tensor), (Tensor, Tensor)](
      estimator, tfi.stack(norm_tf_data.test_dataset.data.toSeq.map(_._1), axis = 0),
      _buffer_size, nTest)

    val index_times = Tensor(
      (0 until causal_horizon).map(_.toDouble)
    ).reshape(
      Shape(causal_horizon)
    )

    val pred_time_lags_test: Tensor = if(prob_timelags) {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(nTest)).cast(FLOAT64)

    } else predictions._2

    val pred_targets: Tensor = if (mo_flag) {
      val all_preds =
        if (prob_timelags) scalers._2.i(predictions._1)
        else scalers._2.i(predictions._1)

      val repeated_times = tfi.stack(Seq.fill(causal_horizon)(pred_time_lags_test.floor), axis = -1)

      val conv_kernel = repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

      all_preds.multiply(conv_kernel).sum(axes = 1).divide(conv_kernel.sum(axes = 1))
    } else {
      scalers._2(0).i(predictions._1)
    }

    val test_labels = dataSet.test_dataset.data.map(_._2).map(t => dtfutils.toDoubleSeq(t).toSeq).toSeq

    val actual_targets = test_labels.zipWithIndex.map(zi => {
      val (z, index) = zi
      val time_lag = pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

      z(time_lag)
    })

    val reg_metrics = new RegressionMetricsTF(pred_targets, actual_targets)

    //write predictions and ground truth to a csv file

    write_predictions(
      dtfutils.toDoubleSeq(pred_targets).toSeq,
      actual_targets,
      dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
      tf_summary_dir/("scatter_test-"+DateTime.now().toString("YYYY-MM-dd-HH-mm")+".csv"))

    val experiment_config = ExperimentType(mo_flag, prob_timelags, "mode")

    val results = SupervisedModelRun(
      (norm_tf_data, scalers),
      model, estimator, None,
      Some(reg_metrics),
      tf_summary_dir, None,
      Some((pred_targets, pred_time_lags_test))
    )

    val partitioned_data = collated_data.partition(DataPipe(tt_partition))

    ExperimentResult(
      experiment_config,
      partitioned_data.training_dataset,
      partitioned_data.test_dataset,
      results
    )
  }


  /**
    * Train and test a CNN based solar wind prediction architecture.
    *
    * @param collated_data Data set of temporally joined
    *                      image paths and GOES X-Ray fluxes.
    *                      This is generally the output after
    *                      executing [[collate_goes_data_range()]]
    *                      with the relevant parameters.
    * @param tt_partition A function which splits the data set
    *                     into train and test sections, based on
    *                     any Boolean function. If the function
    *                     returns true then the instance falls into
    *                     the training set else the test set
    * @param resample If set to true, the training data is resampled
    *                 to balance the occurrence of high flux and low
    *                 flux events.
    *
    * @param tempdir A working directory where the results will be
    *                archived, defaults to user_home_dir/tmp. The model
    *                checkpoints and other results will be stored inside
    *                another directory created in tempdir.
    * @param results_id The suffix added the results/checkpoints directory name.
    * @param stop_criteria When to stop training.
    * @param arch The neural architecture to train.
    *
    * */
  def run_cdt_experiment_omni_ext(
    collated_data: HELIOS_OMNI_DATA_EXT,
    tt_partition: PATTERN_EXT => Boolean,
    resample: Boolean = false,
    preprocess_image: DataPipe[Image, Image] = identityPipe[Image],
    image_to_bytearr: DataPipe[Image, Array[Byte]] = DataPipe((i: Image) => i.argb.flatten.map(_.toByte)),
    processed_image_size: (Int, Int) = (-1, -1),
    num_channels_image: Int = 4,
    image_history: Int = 0,
    image_history_downsampling: Int = 1)(
    results_id: String,
    stop_criteria: StopCriteria,
    tempdir: Path = home/"tmp",
    arch: Layer[(Output, Output), (Output, Output)],
    lossFunc: Layer[((Output, Output), Output), Output],
    mo_flag: Boolean = true,
    prob_timelags: Boolean = true,
    optimizer: Optimizer = tf.train.AdaDelta(0.001),
    miniBatchSize: Int = 16,
    reuseExistingModel: Boolean = false) = {

    //The directories to write model parameters and summaries.
    val resDirName = if(reuseExistingModel) results_id else "helios_omni_hist_"+results_id

    val tf_summary_dir = tempdir/resDirName

    val num_outputs = collated_data.data.head._2._2._2.length

    val causal_horizon = num_outputs

    val size_history = collated_data.data.head._2._2._1.length

    /*
    * After data has been joined/collated,
    * start loading it into tensors
    *
    * */

    val load_image_into_tensor = data.read_image >
      preprocess_image >
      image_to_bytearr >
      /*data.image_to_tensor(
        processed_image_size._1,
        num_channels_image)*/
      DataPipe((arr: Array[Byte]) => dtf.tensor_from_buffer(
        dtype = "UINT8", processed_image_size._1,
        processed_image_size._2,
        num_channels_image)(arr))

    val load_targets_into_tensor = DataPipe((arr: Seq[Double]) => dtf.tensor_f32(num_outputs)(arr:_*))

    val load_targets_hist_into_tensor = DataPipe((arr: Seq[Double]) => dtf.tensor_f32(size_history)(arr:_*))

    val dataSet: TF_DATA_EXT = helios.data.prepare_helios_ts_data_set(
      collated_data,
      load_image_into_tensor,
      load_targets_into_tensor,
      load_targets_hist_into_tensor,
      tt_partition,
      resample,
      image_history,
      image_history_downsampling)


    val (norm_tf_data, scalers): SC_TF_DATA_EXT = scale_helios_dataset_ext(dataSet)

    val data_shapes = (
      (
        Shape(processed_image_size._1, processed_image_size._2, num_channels_image*(image_history_downsampling + 1)),
        Shape(size_history)
      ),
      Shape(causal_horizon)
    )

    val trainData =
      norm_tf_data.training_dataset.build[
        ((Tensor, Tensor), Tensor), ((Output, Output), Output),
        ((DataType.Aux[UByte], DataType.Aux[Float]), DataType.Aux[Float]),
        ((DataType, DataType), DataType),
        ((Shape, Shape), Shape)](
        Left(identityPipe[((Tensor, Tensor), Tensor)]),
        ((UINT8, FLOAT32), FLOAT32),
        data_shapes)
        .repeat()
        .shuffle(1000)
        .batch(miniBatchSize)
        .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")

    val input = tf.learn.Input[
      (Tensor, Tensor), (Output, Output),
      (DataType.Aux[UByte], DataType.Aux[Float]),
      (DataType, DataType), (Shape, Shape)](
      (UINT8, FLOAT32),
      (
        Shape(-1) ++ data_shapes._1._1,
        Shape(-1) ++ data_shapes._1._2
      )
    )


    val trainInput = tf.learn.Input[
      Tensor, Output, DataType.Aux[Float],
      DataType, Shape](FLOAT32, Shape(-1, causal_horizon))

    val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

    val loss = lossFunc >> tf.learn.ScalarSummary("Loss", "ModelLoss")

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    //Now create the model
    val (model, estimator) = dtflearn.build_tf_model(
      arch, input, trainInput, trainingInputLayer,
      loss, optimizer, summariesDir,
      stop_criteria)(trainData)


    val nTest = norm_tf_data.test_dataset.size

    val predictions: (Tensor, Tensor) = dtfutils.buffered_preds[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      Tensor, Output, DataType, Shape, Output,
      (Tensor, Tensor), (Tensor, Tensor), (Tensor, Tensor)](
      estimator,
      (
        tfi.stack(norm_tf_data.test_dataset.data.toSeq.map(_._1._1), axis = 0),
        tfi.stack(norm_tf_data.test_dataset.data.toSeq.map(_._1._2), axis = 0)
      ),
      _buffer_size, nTest)

    val index_times = Tensor(
      (0 until causal_horizon).map(_.toDouble)
    ).reshape(
      Shape(causal_horizon)
    )


    val pred_time_lags_test: Tensor = if(prob_timelags) {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(nTest)).cast(FLOAT64)

    } else predictions._2

    val pred_targets: Tensor = if (mo_flag) {
      val all_preds =
        if (prob_timelags) scalers._2.i(predictions._1)
        else scalers._2.i(predictions._1)

      val repeated_times = tfi.stack(Seq.fill(causal_horizon)(pred_time_lags_test.floor), axis = -1)

      val conv_kernel = repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

      all_preds.multiply(conv_kernel).sum(axes = 1).divide(conv_kernel.sum(axes = 1))
    } else {
      scalers._2(0).i(predictions._1)
    }

    val test_labels = dataSet.test_dataset.data.map(_._2).map(t => dtfutils.toDoubleSeq(t).toSeq).toSeq

    val actual_targets = test_labels.zipWithIndex.map(zi => {
      val (z, index) = zi
      val time_lag = pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

      z(time_lag)
    })

    val reg_metrics = new RegressionMetricsTF(pred_targets, actual_targets)

    //write predictions and ground truth to a csv file

    write_predictions(
      dtfutils.toDoubleSeq(pred_targets).toSeq,
      actual_targets,
      dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
      tf_summary_dir/("scatter_test-"+DateTime.now().toString("YYYY-MM-dd-HH-mm")+".csv"))


    (model, estimator, reg_metrics, tf_summary_dir, scalers, collated_data, norm_tf_data)
  }

  /**
    * Train and test a CNN based solar wind prediction architecture.
    *
    * @param collated_data Data set of temporally joined
    *                      image paths and GOES X-Ray fluxes.
    *                      This is generally the output after
    *                      executing [[collate_goes_data_range()]]
    *                      with the relevant parameters.
    * @param tt_partition A function which splits the data set
    *                     into train and test sections, based on
    *                     any Boolean function. If the function
    *                     returns true then the instance falls into
    *                     the training set else the test set
    * @param resample If set to true, the training data is resampled
    *                 to balance the occurrence of high flux and low
    *                 flux events.
    *
    * @param tempdir A working directory where the results will be
    *                archived, defaults to user_home_dir/tmp. The model
    *                checkpoints and other results will be stored inside
    *                another directory created in tempdir.
    * @param results_id The suffix added the results/checkpoints directory name.
    * @param stop_criteria When to stop training, an instance of [[StopCriteria]]
    * @param arch The neural architecture to train, for example see [[learn.cnn_sw_v1]]
    *
    * */
  def run_cdt_experiment_mc_omni(
    image_sources: Seq[SolarImagesSource],
    collated_data: HELIOS_MC_OMNI_DATA,
    tt_partition: MC_PATTERN => Boolean,
    resample: Boolean = false,
    preprocess_image: Map[SolarImagesSource, DataPipe[Image, Image]],
    images_to_bytes: Map[SolarImagesSource, DataPipe[Image, Array[Byte]]],
    processed_image_size: (Int, Int) = (-1, -1),
    num_channels_image: Int = 4,
    image_history: Int = 0,
    image_history_downsampling: Int = 1)(
    results_id: String,
    stop_criteria: StopCriteria,
    tempdir: Path = home/"tmp",
    arch: Layer[Output, (Output, Output)],
    lossFunc: Layer[((Output, Output), Output), Output],
    mo_flag: Boolean = true,
    prob_timelags: Boolean = true,
    optimizer: Optimizer = tf.train.AdaDelta(0.001),
    miniBatchSize: Int = 16,
    reuseExistingModel: Boolean = false): ExperimentResult[DataSet[MC_PATTERN], Tensor, Tensor, (Tensor, Tensor), (Output, Output)] = {

    //The directories to write model parameters and summaries.
    val resDirName = if(reuseExistingModel) results_id else "helios_omni_"+results_id

    val tf_summary_dir = tempdir/resDirName

    val num_outputs = collated_data.data.head._2._2.length

    /*
    * Create the data processing pipe for processing each image.
    *
    * 1. Load the image into a scrimage object
    * 2. Apply pre-processing filters on the image
    * 3. Convert the processed image into a byte array
    * 4. Load the byte array into a tensor
    * */

    val bytes_to_tensor = DataPipe((arr: Array[Byte]) => dtf.tensor_from_buffer(
      dtype = "UINT8", processed_image_size._1,
      processed_image_size._2,
      num_channels_image)(arr))//data.image_to_tensor(processed_image_size._1, num_channels_image)

    /**/


    val load_image_into_tensor: DataPipe[Map[SolarImagesSource, Seq[Path]], Option[Tensor]] = DataPipe(mc_image => {

      val bytes = mc_image.toSeq.sortBy(_._1.toString).map(kv => {

        val first_non_corrupted_file: (Int, Path) =
          kv._2
            .map(p => (available_bytes(p), p))
            .sortBy(_._1)
            .reverse
            .head

        val processing_for_channel = data.read_image > preprocess_image(kv._1) > images_to_bytes(kv._1)

        if (first_non_corrupted_file._1 > 0) Some(processing_for_channel(first_non_corrupted_file._2)) else None

      }).toArray


      if(bytes.forall(_.isDefined)) Some(bytes_to_tensor(bytes.flatMap(_.get))) else None

    })

    val load_targets_into_tensor = DataPipe((arr: Seq[Double]) => dtf.tensor_f32(num_outputs)(arr:_*))

    val dataSet: TF_DATA = helios.data.prepare_mc_helios_data_set(
      image_sources,
      collated_data,
      load_image_into_tensor,
      load_targets_into_tensor,
      tt_partition,
      resample,
      image_history,
      image_history_downsampling)

    val (norm_tf_data, scalers): SC_TF_DATA = scale_helios_dataset(dataSet)

    val causal_horizon = collated_data.data.head._2._2.length

    val data_shapes = (
      Shape(processed_image_size._1, processed_image_size._2, num_channels_image*(image_history_downsampling + 1)),
      Shape(causal_horizon)
    )

    val trainData = norm_tf_data.training_dataset.build[
      (Tensor, Tensor), (Output, Output),
      (DataType.Aux[UByte], DataType.Aux[Float]), (DataType, DataType),
      (Shape, Shape)](
      Left(identityPipe[(Tensor, Tensor)]),
      dataType = (UINT8, FLOAT32), data_shapes)
      .repeat()
      .shuffle(1000)
      .batch(miniBatchSize)
      .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")
    val input = tf.learn.Input(
      UINT8, Shape(-1) ++ data_shapes._1
    )

    val trainInput = tf.learn.Input(FLOAT32, Shape(-1) ++ data_shapes._2)

    val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

    val loss = lossFunc >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    //Now create the model
    val (model, estimator) = dtflearn.build_tf_model(
      arch, input, trainInput, trainingInputLayer,
      loss, optimizer, java.nio.file.Paths.get(tf_summary_dir.toString()),
      stop_criteria)(
      trainData)

    val nTest = norm_tf_data.test_dataset.size

    val predictions: (Tensor, Tensor) = dtfutils.buffered_preds[
      Tensor, Output, DataType, Shape, (Output, Output),
      Tensor, Output, DataType, Shape, Output,
      Tensor, (Tensor, Tensor), (Tensor, Tensor)](
      estimator, tfi.stack(norm_tf_data.test_dataset.data.toSeq.map(_._1), axis = 0),
      _buffer_size, nTest)

    val index_times = Tensor(
      (0 until causal_horizon).map(_.toDouble)
    ).reshape(
      Shape(causal_horizon)
    )

    val pred_time_lags_test: Tensor = if(prob_timelags) {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(nTest)).cast(FLOAT64)

    } else predictions._2

    val pred_targets: Tensor = if (mo_flag) {
      val all_preds =
        if (prob_timelags) scalers._2.i(predictions._1)
        else scalers._2.i(predictions._1)

      val repeated_times = tfi.stack(Seq.fill(causal_horizon)(pred_time_lags_test.floor), axis = -1)

      val conv_kernel = repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

      all_preds.multiply(conv_kernel).sum(axes = 1).divide(conv_kernel.sum(axes = 1))
    } else {
      scalers._2(0).i(predictions._1)
    }

    val test_labels = dataSet.test_dataset.data.map(_._2).map(t => dtfutils.toDoubleSeq(t).toSeq).toSeq

    val actual_targets = test_labels.zipWithIndex.map(zi => {
      val (z, index) = zi
      val time_lag = pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

      z(time_lag)
    })

    val reg_metrics = new RegressionMetricsTF(pred_targets, actual_targets)

    //write predictions and ground truth to a csv file

    write_predictions(
      dtfutils.toDoubleSeq(pred_targets).toSeq,
      actual_targets,
      dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
      tf_summary_dir/("scatter_test-"+DateTime.now().toString("YYYY-MM-dd-HH-mm")+".csv"))

    val experiment_config = ExperimentType(mo_flag, prob_timelags, "mode")

    val results = SupervisedModelRun(
      (norm_tf_data, scalers),
      model, estimator, None,
      Some(reg_metrics),
      tf_summary_dir, None,
      Some((pred_targets, pred_time_lags_test))
    )

    val partitioned_data = collated_data.partition(DataPipe(tt_partition))

    ExperimentResult(
      experiment_config,
      partitioned_data.training_dataset,
      partitioned_data.test_dataset,
      results
    )
  }



  /**
    * Train and test a CNN based solar wind prediction architecture.
    *
    * @param collated_data Data set of temporally joined
    *                      image paths and GOES X-Ray fluxes.
    *                      This is generally the output after
    *                      executing [[collate_goes_data_range()]]
    *                      with the relevant parameters.
    * @param tt_partition A function which splits the data set
    *                     into train and test sections, based on
    *                     any Boolean function. If the function
    *                     returns true then the instance falls into
    *                     the training set else the test set
    * @param resample If set to true, the training data is resampled
    *                 to balance the occurrence of high flux and low
    *                 flux events.
    *
    * @param tempdir A working directory where the results will be
    *                archived, defaults to user_home_dir/tmp. The model
    *                checkpoints and other results will be stored inside
    *                another directory created in tempdir.
    * @param results_id The suffix added the results/checkpoints directory name.
    * @param stop_criteria Criteria to stop training.
    * @param arch The neural architecture to train.
    *
    * */
  def run_cdt_experiment_mc_omni_ext(
    image_sources: Seq[SOHO],
    collated_data: HELIOS_MC_OMNI_DATA_EXT,
    tt_partition: MC_PATTERN_EXT => Boolean,
    resample: Boolean = false,
    image_pre_process: Map[SOHO, DataPipe[Image, Image]],
    images_to_bytes: DataPipe[Seq[Image], Array[Byte]],
    num_channels_image: Int = 4)(
    results_id: String,
    stop_criteria: StopCriteria = dtflearn.max_iter_stop(5000),
    tempdir: Path = home/"tmp",
    arch: Layer[(Output, Output), (Output, Output)],
    lossFunc: Layer[((Output, Output), Output), Output],
    mo_flag: Boolean = true,
    prob_timelags: Boolean = true,
    optimizer: Optimizer = tf.train.AdaDelta(0.001),
    miniBatchSize: Int = 16,
    inMemoryModel: Boolean = false) = {

    val resDirName = "helios_omni_"+results_id

    val tf_summary_dir = tempdir/resDirName


    /*
    * After data has been joined/collated,
    * start loading it into tensors
    *
    * */
    val dataSet: TF_MC_DATA_EXT = helios.data.prepare_mc_helios_ts_data_set(
      image_sources,
      collated_data,
      tt_partition,
      image_pre_process,
      images_to_bytes,
      resample)


    val (norm_tf_data, scalers): SC_TF_MC_DATA_EXT =
      scale_helios_dataset_mc_ext(dataSet)

    val trainData =
      norm_tf_data.training_data[
        (Output, Output), (DataType, DataType), (Shape, Shape),
        Output, DataType, Shape]
        .repeat()
        .shuffle(10000)
        .batch(miniBatchSize)
        .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")

    val input = tf.learn.Input[
      (Tensor, Tensor), (Output, Output),
      (DataType.Aux[UByte], DataType.Aux[Double]),
      (DataType, DataType), (Shape, Shape)](
      (UINT8, FLOAT64),
      (
        Shape(
          -1,
          dataSet.trainData._1.shape(1),
          dataSet.trainData._1.shape(2),
          dataSet.trainData._1.shape(3)),
        Shape(
          -1,
          dataSet.trainData._2.shape(1))
      )
    )

    val causal_horizon = collated_data.head._2._2._2.length

    val trainInput = tf.learn.Input(FLOAT64, Shape(-1, causal_horizon))

    val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

    val loss = lossFunc >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    //Now create the model
    val (model, estimator) = dtflearn.build_tf_model(
      arch, input, trainInput, trainingInputLayer,
      loss, optimizer, summariesDir, stop_criteria)(
      trainData, inMemory = inMemoryModel)

    val predictions: (Tensor, Tensor) = dtfutils.predict_data[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      Tensor, Output, DataType, Shape, Output,
      (Tensor, Tensor), (Tensor, Tensor)
      ](estimator, dataSet, (false, true), _buffer_size)._2.get

    val index_times = Tensor(
      (0 until causal_horizon).map(_.toDouble)
    ).reshape(
      Shape(causal_horizon)
    )


    val pred_time_lags_test = if(prob_timelags) {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(dataSet.nTest)).cast(FLOAT64)

    } else predictions._2


    val pred_targets: Tensor = if (mo_flag) {
      val all_preds =
        if (prob_timelags) scalers._2.i(predictions._1)
        else scalers._2.i(predictions._1)

      val repeated_times = tfi.stack(Seq.fill(causal_horizon)(pred_time_lags_test.floor), axis = -1)

      val conv_kernel = repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

      all_preds.multiply(conv_kernel).sum(axes = 1).divide(conv_kernel.sum(axes = 1))
    } else {
      scalers._2(0).i(predictions._1)
    }

    val actual_targets = (0 until dataSet.nTest).map(n => {
      val time_lag = pred_time_lags_test(n).scalar.asInstanceOf[Double].toInt
      dataSet.testLabels(n, time_lag).scalar.asInstanceOf[Double]
    })


    val reg_metrics = new RegressionMetricsTF(pred_targets, actual_targets)

    write_predictions(
      dtfutils.toDoubleSeq(pred_targets).toSeq,
      actual_targets,
      dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
      tf_summary_dir/"scatter_test.csv")

    (model, estimator, reg_metrics, tf_summary_dir, scalers, collated_data, norm_tf_data)
  }

  /*def run_experiment_omni_dynamic_time_scales(
    collated_data: Stream[PATTERN],
    tt_partition: PATTERN => Boolean,
    resample: Boolean = false)(
    results_id: String, max_iterations: Int,
    tempdir: Path = home/"tmp",
    arch: Layer[Output, (Output, Output)]) = {

    val resDirName = "helios_omni_"+results_id

    val tf_summary_dir = tempdir/resDirName

    val checkpoints =
      if (exists! tf_summary_dir) ls! tf_summary_dir |? (_.isFile) |? (_.segments.last.contains("model.ckpt-"))
      else Seq()

    val checkpoint_max =
      if(checkpoints.isEmpty) 0
      else (checkpoints | (_.segments.last.split("-").last.split('.').head.toInt)).max

    val iterations = if(max_iterations > checkpoint_max) max_iterations - checkpoint_max else 0


    /*
    * After data has been joined/collated,
    * start loading it into tensors
    *
    * */

    val dataSet = helios.data.create_helios_data_set(
      collated_data,
      tt_partition,
      image_process = DataPipe((i: Image) => i.copy.scale(1.0/math.pow(2.0, 2.0))),
      DataPipe((i: Image) => i.argb.flatten.map(_.toByte)),
      4, resample)

    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainData)

    val train_labels = dataSet.trainLabels

    val labels_mean = dataSet.trainLabels.mean(axes = Tensor(0))

    val labels_stddev = dataSet.trainLabels.subtract(labels_mean).square.mean(axes = Tensor(0)).sqrt

    val norm_train_labels = train_labels.subtract(labels_mean).divide(labels_stddev)

    val trainLabels = tf.data.TensorSlicesDataset(norm_train_labels)

    val trainData =
      trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(64)
        .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")
    val input = tf.learn.Input(
      UINT8,
      Shape(
        -1,
        dataSet.trainData.shape(1),
        dataSet.trainData.shape(2),
        dataSet.trainData.shape(3))
    )

    val num_outputs = collated_data.head._2._2.length

    val trainInput = tf.learn.Input(FLOAT64, Shape(-1, num_outputs))

    val trainingInputLayer = tf.learn.Cast("TrainInput", INT64)

    val lossFunc = DynamicRBFSWLoss("Loss/DynamicRBFWeightedL2", num_outputs)

    val loss = lossFunc >>
      tf.learn.Mean("Loss/Mean") >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    val optimizer = tf.train.AdaGrad(0.002)

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    //Now create the model
    val (model, estimator) = dtflearn.build_tf_model(
      arch, input, trainInput, trainingInputLayer,
      loss, optimizer, summariesDir,
      dtflearn.max_iter_stop(iterations))(
      trainData)

    val predictions: (Tensor, Tensor) = estimator.infer(() => dataSet.testData)

    val pred_targets = predictions._1
      .multiply(labels_stddev(0))
      .add(labels_mean(0))

    val pred_time_lags = predictions._2(::, 1)

    val pred_time_scales = predictions._2(::, 2)

    val metrics = new HeliosOmniTSMetrics(
      dtf.stack(Seq(pred_targets, pred_time_lags), axis = 1), dataSet.testLabels,
      dataSet.testLabels.shape(1),
      pred_time_scales
    )

    (model, estimator, metrics, tf_summary_dir, labels_mean, labels_stddev, collated_data)
  }*/


}
