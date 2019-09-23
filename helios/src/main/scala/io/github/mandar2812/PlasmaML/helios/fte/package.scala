package io.github.mandar2812.PlasmaML.helios

import ammonite.ops._
import org.joda.time._
import breeze.stats._
import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils.{
  GaussianScaler,
  MinMaxScaler,
  ProbitScaler
}
import io.github.mandar2812.dynaml.evaluation._
import io.github.mandar2812.dynaml.optimization._
import io.github.mandar2812.dynaml.tensorflow.evaluation.RegressionMetricsTF
import io.github.mandar2812.dynaml.tensorflow.models.{TFModel, TunableTFModel}
import io.github.mandar2812.dynaml.tensorflow.data._
import io.github.mandar2812.dynaml.tensorflow.utils._
import io.github.mandar2812.dynaml.tensorflow.{
  dtf,
  dtfdata,
  dtflearn,
  dtfutils,
  dtfpipe
}
import io.github.mandar2812.dynaml.tensorflow.implicits._
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.DynaMLPipe._
import _root_.io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{
  L2Regularization,
  L1Regularization
}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.io.github.mandar2812.PlasmaML.helios.fte.data._
import breeze.stats.distributions.ContinuousDistr
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer}
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.variables.RandomNormalInitializer
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow.models.TunableTFModel.HyperParams

package object fte {

  type ModelRunTuning[T, M] = helios.TunedModelRun2[
    T,
    DenseVector[Double],
    Double,
    Output[Double],
    (Output[Double], Output[Double]),
    Double,
    Tensor[Double],
    FLOAT64,
    Shape,
    FLOAT64,
    (Tensor[Double], Tensor[Double]),
    (FLOAT64, FLOAT64),
    (Shape, Shape),
    M,
    (Seq[Double], Seq[Double])
  ]

  type ModelRunTuningSO[T, M] = helios.TunedModelRun2[
    T,
    DenseVector[Double],
    Double,
    Output[Double],
    Output[Double],
    Double,
    Tensor[Double],
    FLOAT64,
    Shape,
    FLOAT64,
    Tensor[Double],
    FLOAT64,
    Shape,
    M,
    Tensor[Double]
  ]

  //Customized layer based on Bala et. al
  val quadratic_fit = (name: String) =>
    new Layer[Output[Double], Output[Double]](name) {
      override val layerType = s"LocalQuadraticFit"

      override def forwardWithoutContext(
        input: Output[Double]
      )(
        implicit mode: Mode
      ) = {

        val aa = tf.variable[Double]("aa", Shape(), RandomNormalInitializer())
        val bb = tf.variable[Double]("bb", Shape(), RandomNormalInitializer())
        val cc = tf.variable[Double]("cc", Shape(), RandomNormalInitializer())

        val a = input.pow(2.0).multiply(aa)
        val b = input.multiply(bb)

        a + b + cc
      }
    }

  /**
    * A configuration object for running experiments
    * on the FTE data sets.
    *
    * Contains cached values of experiment parameters, and data sets.
    * */
  object FTExperiment {

    var config = FteOmniConfig(
      FTEConfig((0, 0), 0, 1, 90d, log_scale_fte = false),
      OMNIConfig((0, 0), log_flag = false)
    )
  }

  type LG = dtflearn.tunable_tf_model.HyperParams => Layer[
    ((Output[Double], Output[Double]), Output[Double]),
    Output[Double]
  ]

  type LG1 = dtflearn.tunable_tf_model.HyperParams => Layer[
    (Output[Double], Output[Double]),
    Output[Double]
  ]

  /**
    * Train the PDT alternating learning algorithm.
    *
    * @param dataset A train-test data set, instance of [[helios.data.TF_DATA_T]].
    * @param tf_summary_dir The directory which stores model runs, parameters, data sets and predictions.
    * @param experiment_config Stores the experiment configuration, see [[data.FteOmniConfig]]
    * @param architechture Neural network model expressed as a Tensorflow Layer.
    * @param hyper_params A list of network and loss function hyper-parameters.
    * @param persistent_hyp_params The hyper-parameters which are not updated during
    *                              the PDT learning procedure. Usually the regularization
    *                              and arhchitecture based hyper-parameters.
    * @param params_to_mutable_params An invertible transformation converting the updatable
    *                                 hyper-parameters into the 'cannonical' PDT hyper-parameters.
    *                                 i.e. &alpha; and &sigma;<sup>2</sup>.
    * @param loss_func_generator A function which takes as input the model hyper-parameters and
    *                            a PDT type loss function. See [[LG]] for type information.
    * @param hyper_prior A prior probability distribution dictating how hyper-parameters are sampled
    *                    during the search procedure. Specified as a scala Map.
    * @param hyp_mapping An optional invertible mapping transformation for the hyper-parameter space.
    *                    Required in case Covariance Matrix Adaption type methods are used for hyper-parameter search.
    * @param iterations The total number of iterations of training to perform after hyper-parameters are chosen.
    * @param iterations_tuning The total number of iterations of training to perform for evaluating each candidate
    *                          model.
    * @param pdt_iterations_tuning The number of PDT updates to perform while evaluating a candidate model.
    * @param pdt_iterations_test The number of PDT updates to perform while training the final chosen model candidate.
    * @param num_samples The number of candidate models to generate during hyper-parameter search. They will be sampled
    *                    from the prior distribution [[hyper_prior]].
    * @param hyper_optimizer A string specifying the hyper-parameter search procedure.
    *                        The following options can be selected:
    *                        <ul>
    *                           <li>"gs": Grid Search/Random Search</li>
    *                           <li>"csa": Coupled Simulated Annealing</li>
    *                           <li>"cma": Covariance Matrix Adaptation</li>
    *                        </ul>
    */
  def run_exp(
    tf_summary_dir: Path,
    architechture: Layer[Output[Double], (Output[Double], Output[Double])],
    hyper_params: List[String],
    persistent_hyp_params: List[String],
    params_to_mutable_params: Encoder[
      dtflearn.tunable_tf_model.HyperParams,
      dtflearn.tunable_tf_model.HyperParams
    ],
    loss_func_generator: LG,
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[
      Double
    ]]],
    hyp_mapping: Option[Map[String, Encoder[Double, Double]]] = None,
    iterations: Int = 150000,
    iterations_tuning: Int = 20000,
    pdt_iterations_tuning: Int = 4,
    pdt_iterations_test: Int = 9,
    num_samples: Int = 20,
    hyper_optimizer: String = "gs",
    miniBatch: Int = 32,
    optimizer: tf.train.Optimizer = tf.train.AdaDelta(0.001f),
    hyp_opt_iterations: Option[Int] = Some(5),
    get_training_preds: Boolean = false,
    existing_exp: Option[Path] = None,
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
      DataPipe[Seq[Tensor[Float]], Double](s =>
          s.map(_.scalar.toDouble).sum / s.length),
    checkpointing_freq: Int = 5,
    log_scale_targets: Boolean = true,
    data_scaling: String = "gauss",
    use_copula: Boolean = false
  ): ModelRunTuning[DenseVector[Double], RegressionMetrics] = {

    require(
      _dataset_serialized(tf_summary_dir),
      s"Data set is not serialized in ${tf_summary_dir}\nPlease run the fte.data.setup_exp_data() method."
    )

    val dataset = {
      println("Using serialized data set")

      val training_data_file = (ls ! tf_summary_dir |? (_.segments.toSeq.last
        .contains("training_data_"))).last
      val test_data_file = (ls ! tf_summary_dir |? (_.segments.toSeq.last
        .contains("test_data_"))).last

      read_data_set[DenseVector[Double], DenseVector[Double]](
        training_data_file,
        test_data_file,
        DataPipe((xs: Array[Double]) => DenseVector(xs)),
        DataPipe((xs: Array[Double]) => DenseVector(xs))
      )
    }

    val causal_window = dataset.training_dataset.data.head._2._2.size

    val data_size = dataset.training_dataset.size

    println("Scaling data attributes")
    val scalers: Either[SCALES, HySCALES] = if (data_scaling == "gauss") {
      println("Performing Gaussian Scaling")
      Left(scale_data(dataset))
    } else {
      println("Performing hybrid gaussian 0-1 Scaling")
      Right(scale_data_hybrid(dataset))
    }

    val scaled_data = {

      val scale_only_targets = scalers match {
        case Left(g_scalers) =>
          if (use_copula) {
            println("Using Gaussian copulas for 0-1 target scaling")
            identityPipe[DateTime] * (
              identityPipe[DenseVector[Double]] * (g_scalers._2 > ProbitScaler)
            )
          } else {
            identityPipe[DateTime] * (
              identityPipe[DenseVector[Double]] * g_scalers._2
            )
          }

        case Right(h_scalers) =>
          println("Performing empirical CDF based scaling of targets")
          identityPipe[DateTime] * (
            identityPipe[DenseVector[Double]] * h_scalers._2
          )
      }

      dataset.copy(
        training_dataset = dataset.training_dataset.map(scale_only_targets)
      )

    }

    val split_data = scaled_data.training_dataset.partition(
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Boolean](
        _ => scala.util.Random.nextDouble() <= 0.7
      )
    )

    val input_shape = Shape(scaled_data.training_dataset.data.head._2._1.size)

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
              dtf.buffer_f64(
                Shape(bufferSize, xdim),
                xs.map(_.toArray).toArray.flatten
              ),
              dtf.buffer_f64(
                Shape(bufferSize, ydim),
                ys.map(_.toArray).toArray.flatten
              )
            )
          }
        )

    val load_input_batch =
      DataPipe[Iterable[DenseVector[Double]], Tensor[Double]](buffer => {
        val bufferSize = buffer.toSeq.length

        val xdim = buffer.head.size
        dtf.buffer_f64(
          Shape(bufferSize, xdim),
          buffer.map(_.toArray).toArray.flatten
        )
      })

    val unzip =
      DataPipe[Iterable[(Tensor[Double], Tensor[Double])], (Iterable[Tensor[Double]], Iterable[Tensor[Double]])](
        _.unzip
      )

    val concatPreds = unzip > duplicate(
      dtfpipe.EagerConcatenate[Double](axis = 0)
    )

    val tf_handle_ops_tuning = dtflearn.model.tf_data_handle_ops[
      (DateTime, (DenseVector[Double], DenseVector[Double])),
      (Tensor[Double], Tensor[Double]),
      (Tensor[Double], Tensor[Double]),
      (Output[Double], Output[Double])
    ](
      bufferSize = 4 * miniBatch,
      patternToTensor = Some(load_pattern_in_tensor),
      caching_mode =
        dtflearn.model.data.FileCache(tf_summary_dir / "data_cache")
    )

    val tf_handle_input = dtflearn.model.tf_data_handle_ops[
      DenseVector[Double],
      Tensor[Double],
      (Tensor[Double], Tensor[Double]),
      Output[Double]
    ](
      bufferSize = 4 * miniBatch,
      patternToTensor = Some(load_input_batch),
      concatOpO = Some(concatPreds)
    )

    val tf_data_ops: dtflearn.model.Ops[(Output[Double], Output[Double])] =
      dtflearn.model.data_ops(
        shuffleBuffer = 10,
        batchSize = miniBatch,
        prefetchSize = 20
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
              miniBatch,
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
            miniBatch,
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
        architechture,
        (FLOAT64, input_shape),
        (FLOAT64, Shape(causal_window)),
        clipGradients = tf.learn.ClipGradientsByNorm(1f)
      )

    val pdtModel = helios.learn.pdt_model(
      causal_window,
      model_function,
      train_config_tuning,
      hyper_params,
      persistent_hyp_params,
      params_to_mutable_params,
      split_data.training_dataset,
      tf_handle_ops_tuning,
      fitness_to_scalar = fitness_to_scalar,
      validation_data = Some(split_data.test_dataset)
    )

    val dt = DateTime.now()

    val run_tuning = () => {
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
            hyper_params,
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

      config
    }

    val config: Map[String, Double] =
      if (exists ! tf_summary_dir / "state.csv") {
        try {
          val best_config: Map[String, Double] = {
            val lines  = read.lines ! tf_summary_dir / "state.csv"
            val keys   = lines.head.split(',')
            val values = lines.last.split(',').map(_.toDouble)
            keys.zip(values).toMap
          }

          println("\nReading from existing best state\n")
          best_config
        } catch {
          case _: Exception => run_tuning()
        }
      } else {
        run_tuning()
      }

    val (best_model, best_config) = pdtModel.build(
      pdt_iterations_test,
      config,
      Some(train_config_test),
      Some(adjusted_iterations / checkpointing_freq)
    )

    val chosen_config = config.filterKeys(persistent_hyp_params.contains) ++ best_config

    write.over(
      tf_summary_dir / "state.csv",
      chosen_config.keys.mkString(start = "", sep = ",", end = "\n") +
        chosen_config.values.mkString(start = "", sep = ",", end = "")
    )

    val extract_tensors =
      tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])]

    val extract_features = tup2_1[DenseVector[Double], DenseVector[Double]]

    val model_predictions_test = best_model.infer_batch(
      scaled_data.test_dataset.map(extract_tensors > extract_features),
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
    ) = scalers match {
      case Left(g_scalers) =>
        process_predictions_bdv2[DenseVector[Double], ReversibleScaler[
          DenseVector[Double]
        ]](
          scaled_data.test_dataset,
          predictions,
          if (use_copula) (g_scalers._1, g_scalers._2 > ProbitScaler)
          else g_scalers,
          causal_window,
          log_scale_omni = log_scale_targets,
          scale_actual_targets = false
        )

      case Right(mm_scalers) =>
        process_predictions_bdv2[DenseVector[Double], ReversibleScaler[
          DenseVector[Double]
        ]](
          scaled_data.test_dataset,
          predictions,
          mm_scalers,
          causal_window,
          log_scale_omni = log_scale_targets,
          scale_actual_targets = false
        )
    }

    val reg_metrics = new RegressionMetrics(
      final_predictions.zip(final_targets).toList,
      final_predictions.length
    )

    val (reg_metrics_train, preds_train) = if (get_training_preds) {

      val model_predictions_train = best_model.infer_batch(
        dataset.training_dataset.map(extract_tensors > extract_features),
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
      ) = scalers match {
        case Left(g_scalers) =>
          process_predictions_bdv2[DenseVector[Double], ReversibleScaler[
            DenseVector[Double]
          ]](
            scaled_data.training_dataset,
            predictions_train,
            if (use_copula) (g_scalers._1, g_scalers._2 > ProbitScaler)
            else g_scalers,
            causal_window,
            log_scale_omni = log_scale_targets,
            scale_actual_targets = true
          )

        case Right(mm_scalers) =>
          process_predictions_bdv2[DenseVector[Double], ReversibleScaler[
            DenseVector[Double]
          ]](
            scaled_data.training_dataset,
            predictions_train,
            mm_scalers,
            causal_window,
            log_scale_omni = log_scale_targets,
            scale_actual_targets = true
          )
      }

      val reg_metrics_train =
        new RegressionMetrics(
          final_predictions_train.zip(final_targets_train).toList,
          final_predictions_train.length
        )

      println("Writing model predictions: Training Data")
      helios.write_predictions[Double](
        unscaled_preds_train,
        predictions_train._2,
        tf_summary_dir,
        "train_" + dt.toString("YYYY-MM-dd-HH-mm")
      )

      helios.write_processed_predictions(
        final_predictions_train,
        final_targets_train,
        pred_time_lags_train,
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

    val results = scalers match {
      case Left(g_scalers) =>
        helios.TunedModelRun2[
          DenseVector[Double],
          DenseVector[Double],
          Double,
          Output[Double],
          (Output[Double], Output[Double]),
          Double,
          Tensor[Double],
          FLOAT64,
          Shape,
          FLOAT64,
          (Tensor[Double], Tensor[Double]),
          (FLOAT64, FLOAT64),
          (Shape, Shape),
          RegressionMetrics,
          (Seq[Double], Seq[Double])
        ](
          (
            scaled_data,
            if (use_copula) (g_scalers._1, g_scalers._2 > ProbitScaler)
            else g_scalers
          ),
          best_model,
          reg_metrics_train,
          Some(reg_metrics),
          tf_summary_dir,
          preds_train,
          Some((final_predictions, pred_time_lags_test))
        )

      case Right(mm_scalers) =>
        helios.TunedModelRun2[
          DenseVector[Double],
          DenseVector[Double],
          Double,
          Output[Double],
          (Output[Double], Output[Double]),
          Double,
          Tensor[Double],
          FLOAT64,
          Shape,
          FLOAT64,
          (Tensor[Double], Tensor[Double]),
          (FLOAT64, FLOAT64),
          (Shape, Shape),
          RegressionMetrics,
          (Seq[Double], Seq[Double])
        ](
          (scaled_data, mm_scalers),
          best_model,
          reg_metrics_train,
          Some(reg_metrics),
          tf_summary_dir,
          preds_train,
          Some((final_predictions, pred_time_lags_test))
        )
    }

    println("Writing model predictions: Test Data")
    helios.write_predictions[Double](
      unscaled_preds_test,
      predictions._2,
      tf_summary_dir,
      "test_" + dt.toString("YYYY-MM-dd-HH-mm")
    )

    helios.write_processed_predictions(
      final_predictions,
      final_targets,
      pred_time_lags_test,
      tf_summary_dir / ("scatter_test-" + dt
        .toString("YYYY-MM-dd-HH-mm") + ".csv")
    )

    println("Writing performance results: Test Data")
    helios.write_performance(
      "test_" + dt.toString("YYYY-MM-dd-HH-mm"),
      reg_metrics,
      tf_summary_dir
    )

    results

  }

  /**
    * Train the PDT alternating learning algorithm
    * for the solar wind prediction task.
    *
    * @param architechture Neural network model expressed as a Tensorflow Layer.
    * @param hyper_params A list of network and loss function hyper-parameters.
    * @param persistent_hyp_params The hyper-parameters which are not updated during
    *                              the PDT learning procedure. Usually the regularization
    *                              and arhchitecture based hyper-parameters.
    * @param params_to_mutable_params An invertible transformation converting the updatable
    *                                 hyper-parameters into the 'cannonical' PDT hyper-parameters.
    *                                 i.e. &alpha; and &sigma;<sup>2</sup>.
    * @param loss_func_generator A function which takes as input the model hyper-parameters and
    *                            a PDT type loss function. See [[LG]] for type information.
    * @param hyper_prior A prior probability distribution dictating how hyper-parameters are sampled
    *                    during the search procedure. Specified as a scala Map.
    * @param hyp_mapping An optional invertible mapping transformation for the hyper-parameter space.
    *                    Required in case Covariance Matrix Adaption type methods are used for hyper-parameter search.
    * @param iterations The total number of iterations of training to perform after hyper-parameters are chosen.
    * @param iterations_tuning The total number of iterations of training to perform for evaluating each candidate
    *                          model.
    * @param pdt_iterations_tuning The number of PDT updates to perform while evaluating a candidate model.
    * @param pdt_iterations_test The number of PDT updates to perform while training the final chosen model candidate.
    * @param num_samples The number of candidate models to generate during hyper-parameter search. They will be sampled
    *                    from the prior distribution [[hyper_prior]].
    * @param hyper_optimizer A string specifying the hyper-parameter search procedure.
    *                        The following options can be selected:
    *                        <ul>
    *                           <li>"gs": Grid Search/Random Search</li>
    *                           <li>"csa": Coupled Simulated Annealing</li>
    *                           <li>"cma": Covariance Matrix Adaptation</li>
    *                        </ul>
    */
  def exp_cdt_alt(
    architechture: Layer[Output[Double], (Output[Double], Output[Double])],
    hyper_params: List[String],
    persistent_hyp_params: List[String],
    params_to_mutable_params: Encoder[
      dtflearn.tunable_tf_model.HyperParams,
      dtflearn.tunable_tf_model.HyperParams
    ],
    loss_func_generator: LG,
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[
      Double
    ]]],
    hyp_mapping: Option[Map[String, Encoder[Double, Double]]] = None,
    iterations: Int = 150000,
    iterations_tuning: Int = 20000,
    pdt_iterations_tuning: Int = 4,
    pdt_iterations_test: Int = 9,
    num_samples: Int = 20,
    hyper_optimizer: String = "gs",
    miniBatch: Int = 32,
    optimizer: tf.train.Optimizer = tf.train.AdaDelta(0.001f),
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    quantity: Int = OMNIData.Quantities.V_SW,
    deltaT: (Int, Int) = (48, 72),
    ts_transform_output: DataPipe[Seq[Double], Seq[Double]] =
      identityPipe[Seq[Double]],
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    latitude_limit: Double = 40d,
    fraction_pca: Double = 0.8,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp,
    hyp_opt_iterations: Option[Int] = Some(5),
    get_training_preds: Boolean = false,
    existing_exp: Option[Path] = None,
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
      DataPipe[Seq[Tensor[Float]], Double](s =>
          s.map(_.scalar.toDouble).sum / s.length),
    checkpointing_freq: Int = 5
  ): helios.Experiment[Double, ModelRunTuning[
    DenseVector[Double],
    RegressionMetrics
  ], FteOmniConfig] = {

    val (experiment_config, tf_summary_dir) = data.setup_exp_data(
      year_range,
      test_year,
      sw_threshold,
      quantity,
      ts_transform_output,
      deltaT,
      deltaTFTE,
      fteStep,
      latitude_limit,
      fraction_pca,
      log_scale_fte,
      log_scale_omni,
      conv_flag,
      fte_data_path,
      summary_top_dir,
      existing_exp
    )

    val results = run_exp(
      tf_summary_dir,
      architechture,
      hyper_params,
      persistent_hyp_params,
      params_to_mutable_params,
      loss_func_generator,
      hyper_prior,
      hyp_mapping,
      iterations,
      iterations_tuning,
      pdt_iterations_tuning,
      pdt_iterations_test,
      num_samples,
      hyper_optimizer,
      miniBatch,
      optimizer,
      hyp_opt_iterations,
      get_training_preds,
      existing_exp,
      fitness_to_scalar,
      checkpointing_freq,
      experiment_config.omni_config.log_flag,
      use_copula = true
    )

    helios.Experiment(
      experiment_config,
      results
    )

  }

  def process_predictions[T](
    scaled_data: DataSet[(DateTime, (T, Tensor[Double]))],
    predictions: (Tensor[Double], Tensor[Double]),
    scalers: (Scaler[T], GaussianScalerTF[Double]),
    causal_window: Int,
    mo_flag: Boolean,
    prob_timelags: Boolean,
    log_scale_omni: Boolean,
    scale_actual_targets: Boolean = true
  ) = {

    val nTest = scaled_data.size

    val index_times = Tensor(
      (0 until causal_window).map(_.toDouble)
    ).reshape(
      Shape(causal_window)
    )

    val pred_time_lags_test: Tensor[Double] = if (prob_timelags) {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(nTest)).castTo[Double]

    } else predictions._2

    val unscaled_preds_test = scalers._2.i(predictions._1)

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
      scalers._2(0).i(predictions._1)
    }

    val test_labels = scaled_data.data
      .map(_._2._2)
      .map(
        t =>
          if (scale_actual_targets) dtfutils.toDoubleSeq(scalers._2.i(t)).toSeq
          else dtfutils.toDoubleSeq(t).toSeq
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

  def process_predictions_bdv[T, S <: ReversibleScaler[DenseVector[Double]]](
    scaled_data: DataSet[(DateTime, (T, DenseVector[Double]))],
    predictions: (Tensor[Double], Tensor[Double]),
    scalers: (Scaler[T], S),
    causal_window: Int,
    log_scale_omni: Boolean,
    scale_actual_targets: Boolean,
    get_tensor_scaler: DataPipe[S, ReversibleScaler[Tensor[Double]]]
  ) = {

    val scaler_targets_tf = get_tensor_scaler(scalers._2)

    val nTest = scaled_data.size

    val index_times = Tensor(
      (0 until causal_window).map(_.toDouble)
    ).reshape(
      Shape(causal_window)
    )

    val pred_time_lags_test: Tensor[Double] = {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(nTest)).castTo[Double]

    }

    val unscaled_preds_test = scaler_targets_tf.i(predictions._1)

    val pred_targets_test: Tensor[Double] = {

      val repeated_times =
        tfi.stack(Seq.fill(causal_window)(pred_time_lags_test.floor), axis = -1)

      val conv_kernel =
        repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

      unscaled_preds_test
        .multiply(conv_kernel)
        .sum(axes = 1)
        .divide(conv_kernel.sum(axes = 1))
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

  def process_predictions_bdv2[T, S <: ReversibleScaler[DenseVector[Double]]](
    scaled_data: DataSet[(DateTime, (T, DenseVector[Double]))],
    predictions: (Tensor[Double], Tensor[Double]),
    scalers: (Scaler[T], S),
    causal_window: Int,
    log_scale_omni: Boolean,
    scale_actual_targets: Boolean
  ): (Seq[Double], Seq[Double], Seq[DenseVector[Double]], Seq[Double]) = {

    val nTest = scaled_data.size

    val index_times = Tensor(
      (0 until causal_window).map(_.toDouble)
    ).reshape(
      Shape(causal_window)
    )

    val pred_time_lags_test: Seq[Double] = dtfutils
      .toDoubleSeq(
        predictions._2.topK(1)._2.reshape(Shape(nTest)).castTo[Double]
      )
      .toSeq

    val unscaled_preds_test = predictions._1
      .unstack(axis = 0)
      .map(t => DenseVector(dtfutils.toDoubleSeq(t).toArray))
      .map(scalers._2.i.run)

    val pred_targets_test: Seq[Double] =
      unscaled_preds_test.zip(pred_time_lags_test).map(pc => pc._1(pc._2.toInt))

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
        pred_time_lags_test(index).toInt

      z(time_lag)
    })

    val (final_predictions, final_targets) =
      if (log_scale_omni)
        (pred_targets_test.map(math.exp), actual_targets.map(math.exp))
      else (pred_targets_test, actual_targets)

    (final_predictions, final_targets, unscaled_preds_test, pred_time_lags_test)
  }

  def exp_single_output_baseline(
    experiment: Path,
    architecture: Layer[Output[Double], Output[Double]],
    hyper_params: List[String],
    loss_func_generator: LG1,
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[
      Double
    ]]],
    fitness_func: Seq[
      DataPipe2[Output[Double], Output[Double], Output[Float]]
    ],
    eval_metric_names: Seq[String] = Seq("mse"),
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
      DataPipe[Seq[Tensor[Float]], Double](s =>
          s.map(_.scalar.toDouble).sum / s.length),
    hyp_mapping: Option[Map[String, Encoder[Double, Double]]] = None,
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    deltaT: Int = 96,
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    data_scaling: String = "gauss",
    use_copula: Boolean = false,
    optimizer: tf.train.Optimizer = tf.train.AdaDelta(0.001f),
    iterations: Int = 50000,
    iterations_tuning: Int = 10000,
    miniBatch: Int = 128,
    hyper_optimizer: String = "gs",
    num_samples: Int = 4,
    hyp_opt_iterations: Option[Int] = Some(5),
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp,
    checkpointing_freq: Int = 1
  ): helios.Experiment[Double, ModelRunTuningSO[
    DenseVector[Double],
    RegressionMetrics
  ], FteOmniConfig] = {

    require(
      _dataset_serialized(experiment),
      s"Data set is not serialized in ${experiment}\nPlease run the fte.data.setup_exp_data() method."
    )

    val experiment_config = read_exp_config(experiment / "config.json").get

    val (dataset, window_size) = {
      println("Using serialized data set")

      val training_data_file = (ls ! experiment |? (_.segments.toSeq.last
        .contains("training_data_"))).last
      val test_data_file = (ls ! experiment |? (_.segments.toSeq.last
        .contains("test_data_"))).last

      val data = read_data_set[DenseVector[Double], DenseVector[Double]](
        training_data_file,
        test_data_file,
        DataPipe((xs: Array[Double]) => DenseVector(xs)),
        DataPipe((xs: Array[Double]) => DenseVector(xs))
      )

      val nT = data.test_dataset.data.head._2._2.length

      val get_window_mid = identityPipe[DateTime] * (
        identityPipe[DenseVector[Double]] * 
        DataPipe((xs: DenseVector[Double]) => {
          val mid: Int = xs.length / 2
          DenseVector(xs(mid))
        })
      )
      
      
      (
        data.copy(
          training_dataset = data.training_dataset.map(get_window_mid),
          test_dataset = data.test_dataset.map(get_window_mid)
        )
      ,
      nT)
    }

    val root_dir = experiment / up

    val dir_name       = s"bs_${experiment.segments.toSeq.last}"
    val tf_summary_dir = root_dir / dir_name

    val input_dim = dataset.training_dataset.data.head._2._1.size

    val num_pred_dims = 1

    println("Scaling data attributes")
    val scalers: Either[SCALES, HySCALES] = if (data_scaling == "gauss") {
      println("Performing Gaussian Scaling")
      Left(scale_data(dataset))
    } else {
      println("Performing hybrid gaussian 0-1 Scaling")
      Right(scale_data_hybrid(dataset))
    }

    val scaled_data = {

      val scale_only_targets = scalers match {
        case Left(g_scalers) =>
          if (use_copula) {
            println("Using Gaussian copulas for 0-1 target scaling")
            identityPipe[DateTime] * (
              identityPipe[DenseVector[Double]] * (g_scalers._2 > ProbitScaler)
            )
          } else {
            identityPipe[DateTime] * (
              identityPipe[DenseVector[Double]] * g_scalers._2
            )
          }

        case Right(h_scalers) =>
          println("Performing empirical CDF based scaling of targets")
          identityPipe[DateTime] * (
            identityPipe[DenseVector[Double]] * h_scalers._2
          )
      }

      dataset.copy(
        training_dataset = dataset.training_dataset.map(scale_only_targets)
      )

    }

    val data_size = scaled_data.training_dataset.size

    val split_data = scaled_data.training_dataset.partition(
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Boolean](
        _ => scala.util.Random.nextDouble() <= 0.7
      )
    )

    val input_shape = Shape(scaled_data.training_dataset.data.head._2._1.size)

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
              dtf.buffer_f64(
                Shape(bufferSize, xdim),
                xs.map(_.toArray).toArray.flatten
              ),
              dtf.buffer_f64(
                Shape(bufferSize, ydim),
                ys.map(_.toArray).toArray.flatten
              )
            )
          }
        )

    val load_input_batch =
      DataPipe[Iterable[DenseVector[Double]], Tensor[Double]](buffer => {
        val bufferSize = buffer.toSeq.length

        val xdim = buffer.head.size
        dtf.buffer_f64(
          Shape(bufferSize, xdim),
          buffer.map(_.toArray).toArray.flatten
        )
      })

    val concatPreds = dtfpipe.EagerConcatenate[Double](axis = 0)

    val tf_handle_ops_tuning = dtflearn.model.tf_data_handle_ops[
      (DateTime, (DenseVector[Double], DenseVector[Double])),
      (Tensor[Double], Tensor[Double]),
      Tensor[Double],
      (Output[Double], Output[Double])
    ](
      bufferSize = 4 * miniBatch,
      patternToTensor = Some(load_pattern_in_tensor),
      caching_mode =
        dtflearn.model.data.FileCache(tf_summary_dir / "data_cache")
    )

    val tf_handle_input = dtflearn.model.tf_data_handle_ops[
      DenseVector[Double],
      Tensor[Double],
      Tensor[Double],
      Output[Double]
    ](
      bufferSize = 4 * miniBatch,
      patternToTensor = Some(load_input_batch),
      concatOpO = Some(concatPreds)
    )

    val tf_data_ops: dtflearn.model.Ops[(Output[Double], Output[Double])] =
      dtflearn.model.data_ops(
        shuffleBuffer = 10,
        batchSize = miniBatch,
        prefetchSize = 10
      )

    val config_to_dir = DataPipe[Map[String, Double], String](
      _.map(kv => s"${kv._1}#${kv._2}").mkString("_")
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
          dtflearn.rel_loss_change_stop(0.005, iterations_tuning)
      ),
      DataPipe(
        (h: Map[String, Double]) =>
          Some(
            timelag.utils.get_train_hooks(
              tf_summary_dir / config_to_dir(h),
              iterations_tuning,
              false,
              data_size,
              miniBatch,
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
        stopCriteria = dtflearn.rel_loss_change_stop(0.005, iterations),
        trainHooks = Some(
          timelag.utils.get_train_hooks(
            tf_summary_dir,
            iterations,
            false,
            data_size,
            miniBatch,
            checkpointing_freq * 2,
            checkpointing_freq
          )
        )
      )

    val tunableTFModel = dtflearn.tunable_tf_model[
      (DateTime, (DenseVector[Double], DenseVector[Double])),
      Output[Double],
      Output[Double],
      Output[Double],
      Double,
      Tensor[Double],
      FLOAT64,
      Shape,
      Tensor[Double],
      FLOAT64,
      Shape,
      Tensor[Double],
      FLOAT64,
      Shape
    ](
      loss_func_generator,
      hyper_params,
      split_data.training_dataset,
      tf_handle_ops_tuning,
      fitness_func,
      architecture,
      (FLOAT64, input_shape),
      (FLOAT64, Shape(1)),
      train_config_tuning,
      fitness_to_scalar = fitness_to_scalar,
      validation_data = Some(split_data.test_dataset),
      inMemory = false
    )

    val gs = hyper_optimizer match {
      case "csa" =>
        new CoupledSimulatedAnnealing[tunableTFModel.type](
          tunableTFModel,
          hyp_mapping
        ).setMaxIterations(
          hyp_opt_iterations.getOrElse(5)
        )

      case "gs" => new GridSearch[tunableTFModel.type](tunableTFModel)

      case "cma" =>
        new CMAES[tunableTFModel.type](
          tunableTFModel,
          hyper_params,
          learning_rate = 0.8,
          hyp_mapping
        ).setMaxIterations(hyp_opt_iterations.getOrElse(5))

      case _ => new GridSearch[tunableTFModel.type](tunableTFModel)
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

    val (_, config) = gs.optimize(hyper_prior.mapValues(_.draw))

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

    write(
      tf_summary_dir / "state.csv",
      config.keys.mkString(start = "", sep = ",", end = "\n") +
        config.values.mkString(start = "", sep = ",", end = "")
    )

    val best_model = tunableTFModel.train_model(
      config,
      Some(train_config_test),
      Some(eval_metric_names.zip(fitness_func)),
      Some(iterations / checkpointing_freq)
    )

    val nTest = dataset.test_dataset.size

    val extract_features = tup2_2[
      DateTime,
      (DenseVector[Double], DenseVector[Double])
    ] > tup2_1[DenseVector[Double], DenseVector[Double]]

    val model_predictions_test = best_model.infer_batch(
      dataset.test_dataset.map(extract_features),
      tf_handle_input
    )

    val targets_scaler: ReversibleScaler[DenseVector[Double]] = scalers match {
      case Left(g_scalers)   => g_scalers._2 > ProbitScaler
      case Right(mm_scalers) => mm_scalers._2
    }

    val convert_tensor_to_breeze = DataPipe(
      (t: Tensor[Double]) => dtfutils.toDoubleSeq(t).toSeq.map(DenseVector(_))
    )

    val pred_targets: Seq[Double] = model_predictions_test match {

      case Left(pred_tensor) =>
        convert_tensor_to_breeze(pred_tensor)
          .map(targets_scaler.i.run)
          .map(_(0))

      case Right(pred_coll) =>
        convert_tensor_to_breeze(
          tfi.concatenate(pred_coll.data.toSeq, axis = 0)
        ).map(targets_scaler.i.run).map(_(0))
    }

    val actual_targets: Seq[Double] =
      dataset.test_dataset.data.toSeq.map(_._2._2(0))

    val reg_metrics = if (log_scale_omni) {
      new RegressionMetrics(
        pred_targets.map(math.exp).zip(actual_targets.map(math.exp)).toList,
        nTest
      )
    } else {
      new RegressionMetrics(pred_targets.zip(actual_targets).toList, nTest)
    }

    val dt = DateTime.now()

    //Dump test predictions
    helios.write_processed_predictions(
      pred_targets,
      actual_targets,
      Seq.fill(pred_targets.length)((window_size/2).toDouble),
      tf_summary_dir / ("scatter_test-" + dt
        .toString("YYYY-MM-dd-HH-mm") + ".csv")
    )

    //Dump performance metrics.
    helios.write_performance(
      "test_" + dt.toString("YYYY-MM-dd-HH-mm"),
      reg_metrics,
      tf_summary_dir
    )

    val results: ModelRunTuningSO[DenseVector[Double], RegressionMetrics] =
      scalers match {
        case Left(g_scalers) =>
          helios.TunedModelRun2(
            (dataset, (g_scalers._1, g_scalers._2 > ProbitScaler)),
            best_model,
            None,
            Some(reg_metrics),
            tf_summary_dir,
            None,
            None
          )

        case Right(mm_scalers) =>
          helios.TunedModelRun2(
            (dataset, mm_scalers),
            best_model,
            None,
            Some(reg_metrics),
            tf_summary_dir,
            None,
            None
          )

      }

    helios.Experiment(
      experiment_config,
      results
    )

  }

  def exp_single_output(
    architecture: Layer[Output[Double], Output[Double]],
    hyper_params: List[String],
    loss_func_generator: LG1,
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[
      Double
    ]]],
    fitness_func: Seq[
      DataPipe2[Output[Double], Output[Double], Output[Float]]
    ],
    eval_metric_names: Seq[String] = Seq("mse"),
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
      DataPipe[Seq[Tensor[Float]], Double](s =>
          s.map(_.scalar.toDouble).sum / s.length),
    hyp_mapping: Option[Map[String, Encoder[Double, Double]]] = None,
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    deltaT: Int = 96,
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    latitude_limit: Double = 40d,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    optimizer: tf.train.Optimizer = tf.train.AdaDelta(0.001f),
    iterations: Int = 50000,
    iterations_tuning: Int = 10000,
    miniBatch: Int = 128,
    hyper_optimizer: String = "gs",
    num_samples: Int = 4,
    hyp_opt_iterations: Option[Int] = Some(5),
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp,
    checkpointing_freq: Int = 1
  ): helios.Experiment[Double, ModelRunTuningSO[
    DenseVector[Double],
    RegressionMetricsTF[Double]
  ], FteOmniConfig] = {

    val sum_dir_prefix = "fte_omni"

    val dt = DateTime.now()

    val summary_dir_index = sum_dir_prefix + s"_so_${deltaT}_" + dt.toString(
      "YYYY-MM-dd-HH-mm"
    )

    val tf_summary_dir = summary_top_dir / summary_dir_index

    val (test_start, test_end) = (
      new DateTime(test_year, 1, 1, 0, 0),
      new DateTime(test_year, 12, 31, 23, 59)
    )

    val (start, end) = (
      new DateTime(year_range.min, 1, 1, 0, 0),
      new DateTime(year_range.max, 12, 31, 23, 59)
    )

    println("\nProcessing FTE Data")

    val fte_data = load_fte_data_bdv(
      fte_data_path,
      carrington_rotations,
      log_scale_fte,
      start,
      end
    )(deltaTFTE, fteStep, latitude_limit, conv_flag)

    val experiment_config = FteOmniConfig(
      FTEConfig(
        (year_range.min, year_range.max),
        deltaTFTE,
        fteStep,
        latitude_limit,
        log_scale_fte
      ),
      OMNIConfig((deltaT, 1), log_scale_omni, OMNIData.Quantities.V_SW),
      multi_output = false,
      probabilistic_time_lags = false,
      timelag_prediction = "none",
      fraction_variance = 1d
    )

    println("Processing OMNI solar wind data")
    val omni_data =
      load_solar_wind_data_bdv2(start, end)((deltaT, 1), log_scale_omni)

    val tt_partition = DataPipe(
      (p: (DateTime, (DenseVector[Double], DenseVector[Double]))) =>
        if (p._1.isAfter(test_start) && p._1.isBefore(test_end))
          false
        else
          true
    )

    println("Constructing joined data set")
    val dataset =
      fte_data
        .join(omni_data)
        .map(
          identityPipe[DateTime] * DataPipe(
            (p: (
              DenseVector[Double],
              (DenseVector[Double], DenseVector[Double])
            )) => (DenseVector.vertcat(p._1, p._2._1), p._2._2)
          )
        )
        .partition(tt_partition)

    val input_dim = dataset.training_dataset.data.head._2._1.size

    val num_pred_dims = 1

    println("Scaling data attributes")
    val scalers = scale_data(dataset)

    val data_size = dataset.training_dataset.size

    val split_data = dataset.training_dataset.partition(
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Boolean](
        _ => scala.util.Random.nextDouble() <= 0.7
      )
    )

    val input_shape = Shape(dataset.training_dataset.data.head._2._1.size)

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
              dtf.buffer_f64(
                Shape(bufferSize, xdim),
                xs.map(_.toArray).toArray.flatten
              ),
              dtf.buffer_f64(
                Shape(bufferSize, ydim),
                ys.map(_.toArray).toArray.flatten
              )
            )
          }
        )

    val load_input_batch =
      DataPipe[Iterable[DenseVector[Double]], Tensor[Double]](buffer => {
        val bufferSize = buffer.toSeq.length

        val xdim = buffer.head.size
        dtf.buffer_f64(
          Shape(bufferSize, xdim),
          buffer.map(_.toArray).toArray.flatten
        )
      })

    val concatPreds = dtfpipe.EagerConcatenate[Double](axis = 0)

    val tf_handle_ops_tuning = dtflearn.model.tf_data_handle_ops[
      (DateTime, (DenseVector[Double], DenseVector[Double])),
      (Tensor[Double], Tensor[Double]),
      Tensor[Double],
      (Output[Double], Output[Double])
    ](
      bufferSize = 4 * miniBatch,
      patternToTensor = Some(load_pattern_in_tensor),
      caching_mode =
        dtflearn.model.data.FileCache(tf_summary_dir / "data_cache")
    )

    val tf_handle_input = dtflearn.model.tf_data_handle_ops[
      DenseVector[Double],
      Tensor[Double],
      Tensor[Double],
      Output[Double]
    ](
      bufferSize = 4 * miniBatch,
      patternToTensor = Some(load_input_batch),
      concatOpO = Some(concatPreds)
    )

    val tf_data_ops: dtflearn.model.Ops[(Output[Double], Output[Double])] =
      dtflearn.model.data_ops(
        shuffleBuffer = 10,
        batchSize = miniBatch,
        prefetchSize = 10
      )

    val config_to_dir = DataPipe[Map[String, Double], String](
      _.map(kv => s"${kv._1}#${kv._2}").mkString("_")
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
          dtflearn.rel_loss_change_stop(0.005, iterations_tuning)
      ),
      DataPipe(
        (h: Map[String, Double]) =>
          Some(
            timelag.utils.get_train_hooks(
              tf_summary_dir / config_to_dir(h),
              iterations_tuning,
              false,
              data_size,
              miniBatch,
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
        stopCriteria = dtflearn.rel_loss_change_stop(0.005, iterations),
        trainHooks = Some(
          timelag.utils.get_train_hooks(
            tf_summary_dir,
            iterations,
            false,
            data_size,
            miniBatch,
            checkpointing_freq * 2,
            checkpointing_freq
          )
        )
      )

    val tunableTFModel = dtflearn.tunable_tf_model[
      (DateTime, (DenseVector[Double], DenseVector[Double])),
      Output[Double],
      Output[Double],
      Output[Double],
      Double,
      Tensor[Double],
      FLOAT64,
      Shape,
      Tensor[Double],
      FLOAT64,
      Shape,
      Tensor[Double],
      FLOAT64,
      Shape
    ](
      loss_func_generator,
      hyper_params,
      split_data.training_dataset,
      tf_handle_ops_tuning,
      fitness_func,
      architecture,
      (FLOAT64, input_shape),
      (FLOAT64, Shape(1)),
      train_config_tuning,
      fitness_to_scalar = fitness_to_scalar,
      validation_data = Some(split_data.test_dataset),
      inMemory = false
    )

    val gs = hyper_optimizer match {
      case "csa" =>
        new CoupledSimulatedAnnealing[tunableTFModel.type](
          tunableTFModel,
          hyp_mapping
        ).setMaxIterations(
          hyp_opt_iterations.getOrElse(5)
        )

      case "gs" => new GridSearch[tunableTFModel.type](tunableTFModel)

      case "cma" =>
        new CMAES[tunableTFModel.type](
          tunableTFModel,
          hyper_params,
          learning_rate = 0.8,
          hyp_mapping
        ).setMaxIterations(hyp_opt_iterations.getOrElse(5))

      case _ => new GridSearch[tunableTFModel.type](tunableTFModel)
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

    val (_, config) = gs.optimize(hyper_prior.mapValues(_.draw))

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

    write(
      tf_summary_dir / "state.csv",
      config.keys.mkString(start = "", sep = ",", end = "\n") +
        config.values.mkString(start = "", sep = ",", end = "")
    )

    val best_model = tunableTFModel.train_model(
      config,
      Some(train_config_test),
      Some(eval_metric_names.zip(fitness_func)),
      Some(iterations / checkpointing_freq)
    )

    val nTest = dataset.test_dataset.size

    val extract_features = tup2_2[
      DateTime,
      (DenseVector[Double], DenseVector[Double])
    ] > tup2_1[DenseVector[Double], DenseVector[Double]]

    val model_predictions_test = best_model.infer_batch(
      dataset.test_dataset.map(extract_features),
      tf_handle_input
    )

    val scaler_targets_tf = GaussianScalerTF(
      dtf.tensor_f64(num_pred_dims)(scalers._2.mean.toArray.toSeq: _*),
      dtf.tensor_f64(num_pred_dims)(scalers._2.sigma.toArray.toSeq: _*)
    )

    val pred_targets = model_predictions_test match {
      case Left(pred_tensor) =>
        scaler_targets_tf.i(pred_tensor).reshape(Shape(nTest))
      case Right(pred_coll) =>
        tfi.stack(pred_coll.map(scaler_targets_tf.i).data.toSeq, axis = 0)
    }

    val stacked_targets = Tensor(
      dataset.test_dataset.data.toSeq.map(_._2._2.toArray.head)
    ).reshape(Shape(nTest))

    val reg_metrics = if (log_scale_omni) {
      new RegressionMetricsTF(pred_targets.exp, stacked_targets.exp)
    } else {
      new RegressionMetricsTF(pred_targets, stacked_targets)
    }

    val results
      : ModelRunTuningSO[DenseVector[Double], RegressionMetricsTF[Double]] =
      helios.TunedModelRun2(
        (dataset, scalers),
        best_model,
        None,
        Some(reg_metrics),
        tf_summary_dir,
        None,
        None
      )

    helios.Experiment(
      experiment_config,
      results
    )

  }

}
