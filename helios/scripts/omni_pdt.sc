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

type DATA   = TFDataSet[(DateTime, (DenseVector[Double], DenseVector[Double]))]
type SCALES = (GaussianScaler, GaussianScaler)

def scale_data(omni_pdt: DATA): SCALES = {
  val (mean_f, sigma_sq_f) = dutils.getStats(
    omni_pdt.training_dataset
      .map(
        tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_1[
          DenseVector[Double],
          DenseVector[Double]
        ]
      )
      .data
  )

  val sigma_f = sqrt(sigma_sq_f)

  val (mean_t, sigma_sq_t) = dutils.getStats(
    omni_pdt.training_dataset
      .map(
        tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_2[
          DenseVector[Double],
          DenseVector[Double]
        ]
      )
      .data
  )

  val sigma_t = sqrt(sigma_sq_t)

  val std_training =
    DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Unit](
      p => {

        //Standardize features
        p._2._1 :-= mean_f
        p._2._1 :/= sigma_f

        //Standardize targets
        p._2._2 :-= mean_t
        p._2._2 :/= sigma_t

      }
    )

  val std_test =
    DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Unit](
      p => {

        //Standardize only features
        p._2._1 :-= mean_f
        p._2._1 :/= sigma_f

      }
    )

  omni_pdt.training_dataset.foreach(std_training)
  omni_pdt.test_dataset.foreach(std_test)
  (GaussianScaler(mean_f, sigma_f), GaussianScaler(mean_t, sigma_t))
}

case class OmniPDTConfig(
  solar_wind_params: List[Int],
  target_quantity: Int,
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
): helios.Experiment[Double, ModelRunTuning, OmniPDTConfig] = {

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

  val hyper_parameters = List(
    "sigma_sq",
    "alpha",
    "reg"
  )

  val persistent_hyper_parameters = List("reg")

  val hyper_prior = Map(
    "reg"      -> UniformRV(-5d, -3d),
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

  type PATTERN = (DateTime, (Tensor[Double], Tensor[Double]))
  val start = new DateTime(start_year, 1, 1, 0, 0, 0)
  val end   = new DateTime(end_year, 12, 31, 23, 59, 59)

  val omni = solar_wind_time_series(start, end, solar_wind_params)
  val omni_ground =
    fte.data
      .load_solar_wind_data_bdv(start, end)(
        causal_window,
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

  val omni_pdt = omni.join(omni_ground).partition(tt_partition)

  val sum_dir_prefix = "omni_pdt_"

  val dt = DateTime.now()

  val mo_flag: Boolean       = true
  val prob_timelags: Boolean = true

  val summary_dir_index = sum_dir_prefix + dt.toString("YYYY-MM-dd-HH-mm")

  val tf_summary_dir = summary_top_dir / summary_dir_index

  val adj_fraction_pca = math.min(math.abs(fraction_pca), 1d)

  val data_size = omni_pdt.training_dataset.size

  println("Scaling data attributes")
  val scalers = scale_data(omni_pdt)

  val input_shape  = Shape(omni_pdt.training_dataset.data.head._2._1.size)
  val output_shape = Shape(omni_pdt.training_dataset.data.head._2._2.size)

  val scalers_tf = (
    //GaussianScalerTF(dtf.tensor_f64(input_shape(0))(scalers._1.mean.toArray:_*), dtf.tensor_f64(input_shape(0))(scalers._1.sigma.toArray:_*)),
    scalers._1,
    GaussianScalerTF(
      dtf.tensor_f64(output_shape(0))(scalers._2.mean.toArray: _*),
      dtf.tensor_f64(output_shape(0))(scalers._2.sigma.toArray: _*)
    )
  )

  val split_data = omni_pdt.training_dataset.partition(
    DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Boolean](
      _ => scala.util.Random.nextDouble() <= 0.7
    )
  )

  val load_pattern_in_tensor =
    IterableDataPipe(
      tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])]
    ) >
      DataPipe[Iterable[(DenseVector[Double], DenseVector[Double])], (Tensor[Double], Tensor[Double])](
        buffer => {
          val (xs, ys)   = buffer.unzip
          val bufferSize = buffer.toSeq.length

          val xdim = xs.head.size
          val ydim = ys.head.size

          (
            dtf.tensor_f64(bufferSize, xdim)(xs.map(_.toArray.toSeq).toSeq.flatten: _*),
            dtf.tensor_f64(bufferSize, ydim)(ys.map(_.toArray.toSeq).toSeq.flatten: _*)
          )
        }
      )

  val load_input_batch =
    DataPipe[Iterable[DenseVector[Double]], Tensor[Double]](buffer => {
      val bufferSize = buffer.toSeq.length

      val xdim = buffer.head.size
      dtf.tensor_f64(bufferSize, xdim)(buffer.map(_.toArray.toSeq).toSeq.flatten: _*)
    })

  val unzip =
    DataPipe[Iterable[(Tensor[Double], Tensor[Double])], (Iterable[Tensor[Double]], Iterable[Tensor[Double]])](
      _.unzip
    )

  val concatPreds = unzip > (dtfpipe.EagerConcatenate[Double](axis = 0) * dtfpipe
    .EagerConcatenate[Double](axis = 0))

  val tf_handle_ops_tuning = dtflearn.model.tf_data_handle_ops[
    (DateTime, (DenseVector[Double], DenseVector[Double])),
    (Tensor[Double], Tensor[Double]),
    (Tensor[Double], Tensor[Double]),
    (Output[Double], Output[Double])
  ](
    bufferSize = 4 * batch_size,
    patternToTensor = Some(load_pattern_in_tensor)
  )

  val tf_handle_ops_test = dtflearn.model.tf_data_handle_ops[
    (DateTime, (DenseVector[Double], DenseVector[Double])),
    (Tensor[Double], Tensor[Double]),
    (Tensor[Double], Tensor[Double]),
    (Output[Double], Output[Double])
  ](
    bufferSize = 4 * batch_size,
    patternToTensor = Some(load_pattern_in_tensor),
    concatOpO = Some(concatPreds)
  )

  val tf_handle_input = dtflearn.model.tf_data_handle_ops[
    DenseVector[Double],
    Tensor[Double],
    (Tensor[Double], Tensor[Double]),
    Output[Double]
  ](
    bufferSize = 4 * batch_size,
    patternToTensor = Some(load_input_batch),
    concatOpO = Some(concatPreds)
  )

  val tf_data_ops: dtflearn.model.Ops[(Output[Double], Output[Double])] =
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
    DataPipe[Map[String, Double], dtflearn.model.Ops[
      (
        Output[Double],
        Output[
          Double
        ]
      )
    ]](_ => tf_data_ops),
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
      (FLOAT64, Shape(causal_window_size)),
      clipGradients = tf.learn.ClipGradientsByNorm(2f)
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

  val extract_tensors = tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])]

  val extract_features = tup2_1[DenseVector[Double], DenseVector[Double]]

  val model_predictions_test = best_model.infer_batch(
    omni_pdt.test_dataset.map(extract_tensors > extract_features),
    tf_handle_input
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
  ) = fte.process_predictions(
    omni_pdt.test_dataset.map(
      identityPipe[DateTime] * (identityPipe[DenseVector[Double]] * DataPipe(
        (y: DenseVector[Double]) => dtf.tensor_f64(y.size)(y.toArray: _*)
      ))
    ),
    predictions,
    scalers_tf,
    causal_window_size,
    mo_flag,
    prob_timelags,
    false,
    false
  )

  val reg_metrics = new RegressionMetricsTF(final_predictions, final_targets)

  val (reg_metrics_train, preds_train) = if (get_training_preds) {

    val model_predictions_train = best_model.infer_batch(
      omni_pdt.training_dataset.map(extract_tensors > extract_features),
      tf_handle_input
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
    ) = fte.process_predictions(
      omni_pdt.training_dataset.map(
        identityPipe[DateTime] * (identityPipe[DenseVector[Double]] * DataPipe(
          (y: DenseVector[Double]) => dtf.tensor_f64(y.size)(y.toArray: _*)
        ))
      ),
      predictions_train,
      scalers_tf,
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
    (omni_pdt, scalers),
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
    OmniPDTConfig(
      solar_wind_params,
      target_quantity,
      (start_year, end_year),
      test_year,
      causal_window,
      fraction_variance = fraction_pca
    ),
    results
  )

}
