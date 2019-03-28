package io.github.mandar2812.PlasmaML.helios

import ammonite.ops._
import org.joda.time._
import breeze.stats._
import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.evaluation._
import io.github.mandar2812.dynaml.optimization._
import io.github.mandar2812.dynaml.models.{TFModel, TunableTFModel}
import io.github.mandar2812.dynaml.tensorflow.data._
import io.github.mandar2812.dynaml.tensorflow.utils._
import io.github.mandar2812.dynaml.tensorflow.{dtf, dtfdata, dtflearn, dtfutils}
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
import _root_.io.github.mandar2812.dynaml.models.TunableTFModel.HyperParams

package object fte {

  type ModelRunTuning = helios.TunedModelRun[
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

    case class FTEConfig(
      data_limits: (Int, Int),
      deltaTFTE: Int,
      fteStep: Int,
      latitude_limit: Double,
      log_scale_fte: Boolean)

    case class OMNIConfig(deltaT: (Int, Int), log_flag: Boolean)

    case class Config(
      fte_config: FTEConfig,
      omni_config: OMNIConfig,
      multi_output: Boolean = true,
      probabilistic_time_lags: Boolean = true,
      timelag_prediction: String = "mode")
        extends helios.Config

    var config = Config(
      FTEConfig((0, 0), 0, 1, 90d, log_scale_fte = false),
      OMNIConfig((0, 0), log_flag = false)
    )

    var fte_data: ZipDataSet[DateTime, Tensor[Double]] =
      dtfdata
        .dataset(Iterable[(DateTime, Tensor[Double])]())
        .to_zip(identityPipe)

    var omni_data: ZipDataSet[DateTime, Tensor[Double]] =
      dtfdata
        .dataset(Iterable[(DateTime, Tensor[Double])]())
        .to_zip(identityPipe)

    def clear_cache(): Unit = {
      fte_data = dtfdata
        .dataset(Iterable[(DateTime, Tensor[Double])]())
        .to_zip(identityPipe)
      omni_data = dtfdata
        .dataset(Iterable[(DateTime, Tensor[Double])]())
        .to_zip(identityPipe)
      config = Config(
        FTEConfig((0, 0), 0, 1, 90d, log_scale_fte = false),
        OMNIConfig((0, 0), log_flag = false)
      )
    }

  }

  def exp_cdt(
    num_neurons: Seq[Int] = Seq(30, 30),
    activation_func: Int => Activation[Double] = (i: Int) =>
      timelag.utils.getReLUAct(1, i),
    optimizer: tf.train.Optimizer = tf.train.Adam(0.001f),
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    deltaT: (Int, Int) = (48, 72),
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    latitude_limit: Double = 40d,
    reg: Double = 0.0001,
    p_wt: Double = 0.75,
    e_wt: Double = 1.0,
    specificity: Double = 1.5,
    temperature: Double = 1.0,
    divergence: helios.learn.cdt_loss.Divergence =
      helios.learn.cdt_loss.KullbackLeibler,
    mo_flag: Boolean = true,
    prob_timelags: Boolean = true,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    iterations: Int = 10000,
    miniBatch: Int = 32,
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp
  ) = {

    val sum_dir_prefix = "fte_omni"

    val dt = DateTime.now()

    val summary_dir_index = {
      if (mo_flag)
        sum_dir_prefix + "_timelag_inference_mo_" + dt.toString(
          "YYYY-MM-dd-HH-mm"
        )
      else
        sum_dir_prefix + "_timelag_inference_" + dt.toString("YYYY-MM-dd-HH-mm")
    }

    val tf_summary_dir = summary_top_dir / summary_dir_index

    val (test_start, test_end) = (
      new DateTime(test_year, 1, 1, 0, 0),
      new DateTime(test_year, 12, 31, 23, 59)
    )

    val (start, end) = (
      new DateTime(year_range.min, 1, 1, 0, 0),
      new DateTime(year_range.max, 12, 31, 23, 59)
    )

    if (FTExperiment.fte_data.size == 0 ||
        FTExperiment.config.fte_config != FTExperiment.FTEConfig(
          (year_range.min, year_range.max),
          deltaTFTE,
          fteStep,
          latitude_limit,
          log_scale_fte
        )) {

      println("\nProcessing FTE Data")

      FTExperiment.fte_data = load_fte_data(
        fte_data_path,
        carrington_rotations,
        log_scale_fte,
        start,
        end
      )(deltaTFTE, fteStep, latitude_limit, conv_flag)

      FTExperiment.config = FTExperiment.config.copy(
        fte_config = FTExperiment.FTEConfig(
          (year_range.min, year_range.max),
          deltaTFTE,
          fteStep,
          latitude_limit,
          log_scale_fte
        )
      )

    } else {
      println("\nUsing cached FTE data sets")
    }

    if (FTExperiment.omni_data.size == 0 ||
        FTExperiment.config.omni_config != FTExperiment.OMNIConfig(
          deltaT,
          log_scale_omni
        )) {

      println("Processing OMNI solar wind data")
      FTExperiment.omni_data =
        load_solar_wind_data(start, end)(deltaT, log_scale_omni)

      FTExperiment.config = FTExperiment.config.copy(
        omni_config = FTExperiment.OMNIConfig(deltaT, log_scale_omni)
      )

    } else {
      println("\nUsing cached OMNI data set")
    }

    val tt_partition = DataPipe(
      (p: (DateTime, (Tensor[Double], Tensor[Double]))) =>
        if (p._1.isAfter(test_start) && p._1.isBefore(test_end))
          false
        else
          true
    )

    println("Constructing joined data set")
    val dataset =
      FTExperiment.fte_data.join(FTExperiment.omni_data).partition(tt_partition)

    val causal_window = dataset.training_dataset.data.head._2._2.shape(0)

    val input_dim = dataset.training_dataset.data.head._2._1.shape

    val input = tf.learn.Input(FLOAT64, Shape(-1) ++ input_dim)

    val trainInput = tf.learn.Input(FLOAT64, Shape(-1, causal_window))

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    val num_pred_dims = timelag.utils.get_num_output_dims(
      causal_window,
      mo_flag,
      prob_timelags,
      "default"
    )

    val (
      net_layer_sizes,
      layer_shapes,
      layer_parameter_names,
      layer_datatypes
    ) =
      timelag.utils.get_ffnet_properties(-1, num_pred_dims, num_neurons)

    val output_mapping = timelag.utils.get_output_mapping[Double](
      causal_window,
      mo_flag,
      prob_timelags,
      "default"
    )

    val filter_depths = Seq(
      Seq(4, 4, 4, 4),
      Seq(2, 2, 2, 2),
      Seq(1, 1, 1, 1)
    )

    val activation = DataPipe[String, Layer[Output[Float], Output[Float]]](
      (s: String) => tf.learn.ReLU(s, 0.01f)
    )

    //Prediction architecture
    val architecture = if (conv_flag) {
      tf.learn.Cast[Double, Float]("Cast/Input") >>
        dtflearn.inception_unit[Float](
          channels = 1,
          num_filters = filter_depths.head,
          activation_generator = activation,
          use_batch_norm = true
        )(layer_index = 1) >>
        tf.learn.MaxPool[Float](
          s"MaxPool_1",
          Seq(1, 3, 3, 1),
          2,
          2,
          SameConvPadding
        ) >>
        dtflearn.inception_unit[Float](
          filter_depths.head.sum,
          filter_depths(1),
          activation,
          use_batch_norm = true
        )(layer_index = 2) >>
        tf.learn.MaxPool[Float](
          s"MaxPool_2",
          Seq(1, 3, 3, 1),
          2,
          2,
          SameConvPadding
        ) >>
        dtflearn.inception_unit[Float](
          filter_depths(1).sum,
          filter_depths.last,
          activation,
          use_batch_norm = true
        )(layer_index = 3) >>
        tf.learn.MaxPool[Float](
          s"MaxPool_3",
          Seq(1, 3, 3, 1),
          2,
          2,
          SameConvPadding
        ) >>
        tf.learn.Flatten[Float]("FlattenFeatures") >>
        tf.learn.Cast[Float, Double]("Cast/Input") >>
        dtflearn.feedforward_stack[Double](activation_func)(
          net_layer_sizes.tail
        ) >>
        output_mapping
    } else {
      dtflearn.feedforward_stack(activation_func)(net_layer_sizes.tail) >>
        output_mapping
    }

    val scope = dtfutils.get_scope(architecture) _

    val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").last))

    val lossFunc = timelag.utils.get_loss[Double, Double, Double](
      causal_window,
      mo_flag,
      prob_timelags,
      prior_wt = p_wt,
      prior_divergence = divergence,
      error_wt = e_wt,
      c = specificity,
      temp = temperature
    )

    val loss = lossFunc >>
      L2Regularization[Double](
        layer_scopes,
        layer_parameter_names,
        layer_datatypes,
        layer_shapes,
        reg
      ) >>
      tf.learn.ScalarSummary[Double]("Loss", "ModelLoss")

    println("Scaling data attributes")
    val (scaled_data, scalers): SC_DATA = scale_dataset[Double].run(
      dataset.copy(
        training_dataset = dataset.training_dataset
          .map((p: (DateTime, (Tensor[Double], Tensor[Double]))) => p._2),
        test_dataset = dataset.test_dataset.map(
          (p: (DateTime, (Tensor[Double], Tensor[Double]))) => p._2
        )
      )
    )

    val train_data_tf = {
      scaled_data.training_dataset
        .build[
          (Tensor[Double], Tensor[Double]),
          (Output[Double], Output[Double]),
          (FLOAT64, FLOAT64),
          (Shape, Shape)
        ](
          identityPipe[(Tensor[Double], Tensor[Double])],
          (FLOAT64, FLOAT64),
          (input_dim, Shape(causal_window))
        )
        .repeat()
        .shuffle(10)
        .batch[(FLOAT64, FLOAT64), (Shape, Shape)](miniBatch)
        .prefetch(10)
    }

    val (model, estimator) = dtflearn.build_tf_model[Output[Double], Output[
      Double
    ], (Output[Double], Output[Double]), (Output[Double], Output[Double]), Double, ((Output[Double], Output[Double]), (Output[Double], Output[Double])), FLOAT64, Shape, FLOAT64, Shape](
      architecture,
      input,
      trainInput,
      loss,
      optimizer,
      summariesDir,
      dtflearn.rel_loss_change_stop(0.005, iterations)
    )(train_data_tf)

    val nTest = scaled_data.test_dataset.size

    val predictions: (Tensor[Double], Tensor[Double]) = dtfutils.buffered_preds[
      Output[Double],
      Output[Double],
      (Output[Double], Output[Double]),
      (Output[Double], Output[Double]),
      Double,
      Tensor[Double],
      FLOAT64,
      Shape,
      (Tensor[Double], Tensor[Double]),
      (FLOAT64, FLOAT64),
      (Shape, Shape),
      Tensor[Double],
      Tensor[Double],
      (Tensor[Double], Tensor[Double])
    ](
      estimator,
      tfi
        .stack[Double](scaled_data.test_dataset.data.toSeq.map(_._1), axis = 0),
      500,
      nTest
    )

    val index_times = Tensor(
      (0 until causal_window).map(_.toDouble)
    ).reshape(
      Shape(causal_window)
    )

    val pred_time_lags_test: Tensor[Double] = if (prob_timelags) {
      val unsc_probs = predictions._2

      unsc_probs.topK(1)._2.reshape(Shape(nTest)).castTo[Double]

    } else predictions._2

    val pred_targets: Tensor[Double] = if (mo_flag) {
      val all_preds =
        if (prob_timelags) scalers._2.i(predictions._1)
        else scalers._2.i(predictions._1)

      val repeated_times =
        tfi.stack(Seq.fill(causal_window)(pred_time_lags_test.floor), axis = -1)

      val conv_kernel =
        repeated_times.subtract(index_times).square.multiply(-1.0).exp.floor

      all_preds
        .multiply(conv_kernel)
        .sum(axes = 1)
        .divide(conv_kernel.sum(axes = 1))
    } else {
      scalers._2(0).i(predictions._1)
    }

    val test_labels = scaled_data.test_dataset.data
      .map(_._2)
      .map(t => dtfutils.toDoubleSeq(t).toSeq)
      .toSeq

    val actual_targets = test_labels.zipWithIndex.map(zi => {
      val (z, index) = zi
      val time_lag =
        pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

      z(time_lag)
    })

    val (final_predictions, final_targets) =
      if (log_scale_omni) (pred_targets.exp, actual_targets.map(math.exp))
      else (pred_targets, actual_targets)

    val reg_metrics = new RegressionMetricsTF(final_predictions, final_targets)

    val experiment_config =
      helios.ExperimentConfig(mo_flag, prob_timelags, "mode")

    val results = helios
      .SupervisedModelRun[Double, Output[Double], (Output[Double], Output[Double]), Double, Tensor[
        Double
      ], (Tensor[Double], Tensor[Double])](
        (scaled_data, scalers),
        model,
        estimator,
        None,
        Some(reg_metrics),
        tf_summary_dir,
        None,
        Some((final_predictions, pred_time_lags_test))
      )

    helios.write_processed_predictions(
      dtfutils.toDoubleSeq(final_predictions).toSeq,
      final_targets,
      dtfutils.toDoubleSeq(pred_time_lags_test).toSeq,
      tf_summary_dir / ("scatter_test-" + DateTime
        .now()
        .toString("YYYY-MM-dd-HH-mm") + ".csv")
    )

    helios.ExperimentResult(
      experiment_config,
      dataset.training_dataset,
      dataset.test_dataset,
      results
    )

  }

  type LG = dtflearn.tunable_tf_model.HyperParams => Layer[
    ((Output[Double], Output[Double]), Output[Double]),
    Output[Double]
  ]

  def exp_cdt_tuning(
    arch: Layer[Output[Double], (Output[Double], Output[Double])],
    hyper_params: List[String],
    loss_func_generator: LG,
    fitness_func: DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[
      Float
    ]],
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[
      Double
    ]]],
    hyp_mapping: Option[Map[String, Encoder[Double, Double]]] = None,
    iterations: Int = 150000,
    iterations_tuning: Int = 20000,
    num_samples: Int = 20,
    hyper_optimizer: String = "gs",
    miniBatch: Int = 32,
    optimizer: tf.train.Optimizer = tf.train.AdaDelta(0.001f),
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    quantity: Int = OMNIData.Quantities.V_SW,
    deltaT: (Int, Int) = (48, 72),
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    latitude_limit: Double = 40d,
    divergence: helios.learn.cdt_loss.Divergence =
      helios.learn.cdt_loss.KullbackLeibler,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp,
    hyp_opt_iterations: Option[Int] = Some(5),
    existing_exp: Option[Path] = None
  ): helios.Experiment[Double, ModelRunTuning, FTExperiment.Config] = {

    val mo_flag: Boolean       = true
    val prob_timelags: Boolean = true

    val sum_dir_prefix = if (conv_flag) "fte_omni_conv" else "fte_omni"

    val dt = DateTime.now()

    val summary_dir_index = {
      if (mo_flag) sum_dir_prefix + "_mo_tl_" + dt.toString("YYYY-MM-dd-HH-mm")
      else sum_dir_prefix + "_tl_" + dt.toString("YYYY-MM-dd-HH-mm")
    }

    val tf_summary_dir =
      existing_exp.getOrElse(summary_top_dir / summary_dir_index)

    val (test_start, test_end) = (
      new DateTime(test_year, 1, 1, 0, 0),
      new DateTime(test_year, 12, 31, 23, 59)
    )

    val (start, end) = (
      new DateTime(year_range.min, 1, 1, 0, 0),
      new DateTime(year_range.max, 12, 31, 23, 59)
    )

    val tt_partition = DataPipe(
      (p: (DateTime, (DenseVector[Double], Tensor[Double]))) =>
        if (p._1.isAfter(test_start) && p._1.isBefore(test_end))
          false
        else
          true
    )

    println("\nProcessing FTE Data")
    val fte_data = load_fte_data_bdv(
      fte_data_path,
      carrington_rotations,
      log_scale_fte,
      start,
      end
    )(deltaTFTE, fteStep, latitude_limit, conv_flag)

    println("Processing OMNI solar wind data")
    val omni_data =
      load_solar_wind_data(start, end)(deltaT, log_scale_omni, quantity)

    println("Constructing joined data set")
    val dataset = fte_data.join(omni_data).partition(tt_partition)

    val causal_window = dataset.training_dataset.data.head._2._2.shape(0)

    val data_size = dataset.training_dataset.size

    val scaling_op = scale_timed_data2[Double](fraction = 0.81)

    println("Scaling data attributes")
    val (scaled_data, scalers): helios.data.SC_TF_DATA_T2[DenseVector[Double], Double] =
      scaling_op.run(dataset)

    val input_shape = Shape(scaled_data.training_dataset.data.head._2._1.size)

    val load_pattern_in_tensor =
      tup2_2[DateTime, (DenseVector[Double], Tensor[Double])] >
        (
          DataPipe(
            (dv: DenseVector[Double]) =>
              dtf.tensor_f64(input_shape(0))(dv.toArray.toSeq: _*)
          ) *
            identityPipe[Tensor[Double]]
        )

    val unzip =
      DataPipe[Iterable[(Tensor[Double], Tensor[Double])], (Iterable[Tensor[Double]], Iterable[Tensor[Double]])](
        _.unzip
      )

    val concatPreds = unzip > (helios.concatOperation[Double](ax = 0) * helios
      .concatOperation[Double](ax = 0))

    val tf_handle_ops_tuning = dtflearn.model.tf_data_handle_ops[
      (DateTime, (DenseVector[Double], Tensor[Double])),
      Tensor[Double],
      Tensor[Double],
      (Tensor[Double], Tensor[Double]),
      Output[Double],
      Output[Double]
    ](
      patternToTensor = Some(load_pattern_in_tensor),
      concatOpI = Some(stackOperation[Double](ax = 0)),
      concatOpT = Some(stackOperation[Double](ax = 0))
    )

    val tf_handle_ops_test = dtflearn.model.tf_data_handle_ops[
      (DateTime, (DenseVector[Double], Tensor[Double])),
      Tensor[Double],
      Tensor[Double],
      (Tensor[Double], Tensor[Double]),
      Output[Double],
      Output[Double]
    ](
      patternToTensor = Some(load_pattern_in_tensor),
      concatOpI = Some(stackOperation[Double](ax = 0)),
      concatOpT = Some(stackOperation[Double](ax = 0)),
      concatOpO = Some(concatPreds)
    )

    val tf_data_ops: dtflearn.model.Ops[Output[Double], Output[Double]] =
      dtflearn.model.data_ops(
        shuffleBuffer = 10,
        batchSize = miniBatch,
        prefetchSize = 10
      )

    val train_config_tuning: MetaPipe[
      Path,
      dtflearn.tunable_tf_model.HyperParams,
      dtflearn.model.Config[Output[Double], Output[Double]]
    ] =
      dtflearn.tunable_tf_model.ModelFunction.hyper_params_to_dir >>
        DataPipe(
          (p: Path) =>
            dtflearn.model.trainConfig(
              p,
              tf_data_ops,
              optimizer,
              dtflearn.rel_loss_change_stop(0.005, iterations_tuning),
              Some(
                dtflearn.model._train_hooks(
                  p,
                  iterations_tuning / 3,
                  iterations_tuning / 3,
                  iterations_tuning
                )
              )
            )
        )

    val checkpoints =
      if (exists ! tf_summary_dir)
        ls ! tf_summary_dir |? (_.isFile) |? (_.segments.last
          .contains("model.ckpt-"))
      else Seq()

    val checkpoint_max =
      if (checkpoints.isEmpty) 0
      else
        (checkpoints | (_.segments.last
          .split("-")
          .last
          .split('.')
          .head
          .toInt)).max

    val max_iterations =
      if (iterations > checkpoint_max) iterations - checkpoint_max else 0

    val train_config_test =
      dtflearn.model.trainConfig[Output[Double], Output[Double]](
        summaryDir = tf_summary_dir,
        tf_data_ops,
        optimizer,
        stopCriteria = dtflearn.rel_loss_change_stop(0.005, max_iterations),
        trainHooks = Some(
          dtflearn.model._train_hooks(
            tf_summary_dir,
            math.max(max_iterations / 3, 1).toInt,
            math.max(max_iterations / 3, 1).toInt,
            math.max(max_iterations / 2, 1).toInt
          )
        )
      )

    val tunableTFModel: TunableTFModel[
      (DateTime, (DenseVector[Double], Tensor[Double])),
      Output[Double],
      Output[Double],
      (Output[Double], Output[Double]),
      Double,
      Tensor[Double],
      FLOAT64,
      Shape,
      Tensor[Double],
      FLOAT64,
      Shape,
      (Tensor[Double], Tensor[Double]),
      (FLOAT64, FLOAT64),
      (Shape, Shape)
    ] =
      dtflearn.tunable_tf_model(
        loss_func_generator,
        hyper_params,
        scaled_data.training_dataset,
        tf_handle_ops_tuning,
        fitness_func,
        arch,
        (FLOAT64, input_shape),
        (FLOAT64, Shape(causal_window)),
        train_config_tuning(tf_summary_dir),
        data_split_func = Some(
          DataPipe[(DateTime, (DenseVector[Double], Tensor[Double])), Boolean](
            _ => scala.util.Random.nextGaussian() <= 0.7
          )
        ),
        inMemory = false
      )

    val run_tuning = () => {
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

      val (_, best_config) = gs.optimize(hyper_prior.mapValues(_.draw))

      println(
        "--------------------------------------------------------------------"
      )
      println("\nModel tuning complete")

      write.over(
        tf_summary_dir / "state.csv",
        best_config.keys.mkString(start = "", sep = ",", end = "\n") +
          best_config.values.mkString(start = "", sep = ",", end = "")
      )

      best_config
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

    println("Chosen configuration:")
    pprint.pprintln(config)
    println(
      "--------------------------------------------------------------------"
    )

    println("Training final model based on chosen configuration")

    val best_model = tunableTFModel.train_model(config, Some(train_config_test))

    val extract_tensors = load_pattern_in_tensor

    val extract_features = tup2_1[Tensor[Double], Tensor[Double]]

    val model_predictions_test = best_model.infer_batch(
      scaled_data.test_dataset.map(extract_tensors > extract_features),
      train_config_test.data_processing,
      tf_handle_ops_test
    )

    val predictions = model_predictions_test match {
      case Left(tensor)      => tensor
      case Right(collection) => timelag.utils.collect_predictions(collection)
    }

    val nTest = scaled_data.test_dataset.size

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

    val pred_targets: Tensor[Double] = if (mo_flag) {

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

    val test_labels = scaled_data.test_dataset.data
      .map(_._2._2)
      .map(t => dtfutils.toDoubleSeq(t).toSeq)
      .toSeq

    val actual_targets = test_labels.zipWithIndex.map(zi => {
      val (z, index) = zi
      val time_lag =
        pred_time_lags_test(index).scalar.asInstanceOf[Double].toInt

      z(time_lag)
    })

    val (final_predictions, final_targets) =
      if (log_scale_omni) (pred_targets.exp, actual_targets.map(math.exp))
      else (pred_targets, actual_targets)

    val reg_metrics = new RegressionMetricsTF(final_predictions, final_targets)

    val experiment_config = FTExperiment.Config(
      FTExperiment.FTEConfig(
        (year_range.min, year_range.max),
        deltaTFTE,
        fteStep,
        latitude_limit,
        log_scale_fte
      ),
      FTExperiment.OMNIConfig(deltaT, log_scale_omni)
    )

    val results = helios.TunedModelRun(
      (scaled_data, scalers),
      best_model,
      None,
      Some(reg_metrics),
      tf_summary_dir,
      None,
      Some((final_predictions, pred_time_lags_test))
    )

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

    helios.Experiment(
      experiment_config,
      results
    )

  }

  def exp_single_output(
    num_neurons: Seq[Int] = Seq(30, 30),
    activation_func: Int => Activation[Double] = timelag.utils.getReLUAct(1, _),
    optimizer: tf.train.Optimizer = tf.train.Adam(0.001f),
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    deltaT: Int = 96,
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    latitude_limit: Double = 40d,
    reg: Double = 0.0001,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    iterations: Int = 10000,
    miniBatch: Int = 32,
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp
  ) = {

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

    if (FTExperiment.fte_data.size == 0 ||
        FTExperiment.config.fte_config != FTExperiment.FTEConfig(
          (year_range.min, year_range.max),
          deltaT,
          fteStep,
          latitude_limit,
          log_scale_fte
        )) {

      println("\nProcessing FTE Data")

      FTExperiment.fte_data = load_fte_data(
        fte_data_path,
        carrington_rotations,
        log_scale_fte,
        start,
        end
      )(deltaTFTE, fteStep, latitude_limit, conv_flag)

      FTExperiment.config = FTExperiment.config.copy(
        fte_config = FTExperiment.FTEConfig(
          (year_range.min, year_range.max),
          deltaTFTE,
          fteStep,
          latitude_limit,
          log_scale_fte
        )
      )

    } else {
      println("\nUsing cached FTE data sets")
    }

    if (FTExperiment.omni_data.size == 0 ||
        FTExperiment.config.omni_config != FTExperiment.OMNIConfig(
          (deltaT, 1),
          log_scale_omni
        )) {

      println("Processing OMNI solar wind data")
      FTExperiment.omni_data =
        load_solar_wind_data(start, end)((deltaT, 1), log_scale_omni)

      FTExperiment.config = FTExperiment.config.copy(
        omni_config = FTExperiment.OMNIConfig((deltaT, 1), log_scale_omni)
      )

    } else {
      println("\nUsing cached OMNI data set")
    }

    val tt_partition = DataPipe(
      (p: (DateTime, (Tensor[Double], Tensor[Double]))) =>
        if (p._1.isAfter(test_start) && p._1.isBefore(test_end))
          false
        else
          true
    )

    println("Constructing joined data set")
    val dataset =
      FTExperiment.fte_data.join(FTExperiment.omni_data).partition(tt_partition)

    val input_dim = dataset.training_dataset.data.head._2._1.shape

    val input = tf.learn.Input(FLOAT64, Shape(-1) ++ input_dim)

    val trainInput = tf.learn.Input(FLOAT64, Shape(-1))

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    val num_pred_dims = 1

    val (
      net_layer_sizes,
      layer_shapes,
      layer_parameter_names,
      layer_datatypes
    ) =
      timelag.utils.get_ffnet_properties(-1, num_pred_dims, num_neurons)

    val filter_depths = Seq(
      Seq(4, 4, 4, 4),
      Seq(2, 2, 2, 2),
      Seq(1, 1, 1, 1)
    )

    val activation = DataPipe[String, Layer[Output[Float], Output[Float]]](
      (s: String) => tf.learn.ReLU(s, 0.01f)
    )

    //Prediction architecture
    val architecture = if (conv_flag) {
      tf.learn.Cast[Double, Float]("Cast/Input") >>
        dtflearn.inception_unit[Float](
          channels = 1,
          num_filters = filter_depths.head,
          activation_generator = activation,
          use_batch_norm = true
        )(layer_index = 1) >>
        tf.learn.MaxPool[Float](
          s"MaxPool_1",
          Seq(1, 3, 3, 1),
          2,
          2,
          SameConvPadding
        ) >>
        dtflearn.inception_unit[Float](
          filter_depths.head.sum,
          filter_depths(1),
          activation,
          use_batch_norm = true
        )(layer_index = 2) >>
        tf.learn.MaxPool[Float](
          s"MaxPool_2",
          Seq(1, 3, 3, 1),
          2,
          2,
          SameConvPadding
        ) >>
        dtflearn.inception_unit[Float](
          filter_depths(1).sum,
          filter_depths.last,
          activation,
          use_batch_norm = true
        )(layer_index = 3) >>
        tf.learn.MaxPool[Float](
          s"MaxPool_3",
          Seq(1, 3, 3, 1),
          2,
          2,
          SameConvPadding
        ) >>
        tf.learn.Flatten[Float]("FlattenFeatures") >>
        tf.learn.Cast[Float, Double]("Cast/Input") >>
        dtflearn.feedforward_stack[Double](activation_func)(
          net_layer_sizes.tail
        )
    } else {
      dtflearn.feedforward_stack(activation_func)(net_layer_sizes.tail)
    }

    val scope = dtfutils.get_scope(architecture) _

    val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").last))

    val loss = tf.learn.L2Loss[Double, Double]("Loss/L2") >>
      tf.learn.ScalarSummary("Loss/Error", "Error") >>
      L2Regularization(
        layer_scopes,
        layer_parameter_names,
        layer_datatypes,
        layer_shapes,
        reg,
        "L2Reg"
      ) >>
      tf.learn.ScalarSummary("Loss/Net", "ModelLoss")

    println("Scaling data attributes")
    val (scaled_data, scalers): SC_DATA = scale_dataset[Double].run(
      dataset.copy(
        training_dataset = dataset.training_dataset
          .map((p: (DateTime, (Tensor[Double], Tensor[Double]))) => p._2),
        test_dataset = dataset.test_dataset.map(
          (p: (DateTime, (Tensor[Double], Tensor[Double]))) => p._2
        )
      )
    )

    val train_data_tf = {
      scaled_data.training_dataset
        .build[
          (Tensor[Double], Tensor[Double]),
          (Output[Double], Output[Double]),
          (FLOAT64, FLOAT64),
          (Shape, Shape)
        ](
          identityPipe[(Tensor[Double], Tensor[Double])],
          (FLOAT64, FLOAT64),
          (input_dim, Shape())
        )
        .repeat()
        .shuffle(10)
        .batch[(FLOAT64, FLOAT64), (Shape, Shape)](miniBatch)
        .prefetch(10)
    }

    val (model, estimator) = dtflearn
      .build_tf_model[Output[Double], Output[Double], Output[Double], Output[
        Double
      ], Double, (Output[Double], (Output[Double], Output[Double])), FLOAT64, Shape, FLOAT64, Shape](
        architecture,
        input,
        trainInput,
        loss,
        optimizer,
        summariesDir,
        dtflearn.rel_loss_change_stop(0.005, iterations)
      )(train_data_tf)

    val nTest = scaled_data.test_dataset.size

    implicit val ev = concatTensorSplits[Double]

    val predictions: Tensor[Double] =
      dtfutils.buffered_preds[Output[Double], Output[Double], Output[Double], Output[
        Double
      ], Double, Tensor[Double], FLOAT64, Shape, Tensor[Double], FLOAT64, Shape, Tensor[
        Double
      ], Tensor[Double], Tensor[Double]](
        estimator,
        tfi.stack(scaled_data.test_dataset.data.toSeq.map(_._1), axis = 0),
        500,
        nTest
      )

    val pred_targets = scalers._2.i(predictions).reshape(Shape(nTest))

    val stacked_targets =
      tfi.stack(scaled_data.test_dataset.data.toSeq.map(_._2), axis = 0)

    val reg_metrics = if (log_scale_omni) {
      new RegressionMetricsTF(pred_targets.exp, stacked_targets.exp)
    } else {
      new RegressionMetricsTF(pred_targets, stacked_targets)
    }

    val experiment_config = helios.ExperimentConfig(
      multi_output = false,
      probabilistic_time_lags = false,
      "single-output"
    )

    val results = helios.SupervisedModelRun(
      (scaled_data, scalers),
      model,
      estimator,
      None,
      Some(reg_metrics),
      tf_summary_dir,
      None,
      Some(pred_targets)
    )

    helios.ExperimentResult(
      experiment_config,
      dataset.training_dataset,
      dataset.test_dataset,
      results
    )

  }

}
