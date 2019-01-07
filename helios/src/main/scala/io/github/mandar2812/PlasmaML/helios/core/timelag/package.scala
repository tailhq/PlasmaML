package io.github.mandar2812.PlasmaML.helios.core

import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import org.joda.time.DateTime
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.data.{DataSet, TFDataSet}
import _root_.io.github.mandar2812.dynaml.tensorflow.utils._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.evaluation._
import _root_.io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GridSearch}
import _root_.io.github.mandar2812.dynaml.models.{TFModel, TunableTFModel}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data.HeliosDataSet
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.{INFERENCE, Mode, SupervisedTrainableModel}
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag.utils._
import breeze.stats.distributions.ContinuousDistr

package object timelag {



  /**
    * A model run contains a tensorflow model/estimator as
    * well as its training/test data set and meta data regarding
    * the training/evaluation process.
    *
    * */
  sealed trait ModelRun {

    type MODEL
    type ESTIMATOR

    type DATA

    val summary_dir: Path

    val data_and_scales: (DATA, (GaussianScalerTF, GaussianScalerTF))

    val metrics_train: (RegressionMetricsTF, RegressionMetricsTF)

    val metrics_test: (RegressionMetricsTF, RegressionMetricsTF)

    val model: MODEL

    val estimator: ESTIMATOR

  }

  case class TunedModelRun(
    data_and_scales: (TFDataSet[(Tensor, Tensor)], (GaussianScalerTF, GaussianScalerTF)),
    model: TFModel[
      Tensor, Output, DataType.Aux[Float], DataType, Shape, (Output, Output), (Tensor, Tensor),
      Tensor, Output, DataType.Aux[Double], DataType, Shape, Output],
    metrics_train: (RegressionMetricsTF, RegressionMetricsTF),
    metrics_test: (RegressionMetricsTF, RegressionMetricsTF),
    summary_dir: Path,
    training_preds: (Tensor, Tensor),
    test_preds: (Tensor, Tensor)) extends ModelRun {

    override type DATA = TFDataSet[(Tensor, Tensor)]

    override type MODEL = TFModel[
      Tensor, Output, DataType.Aux[Float], DataType, Shape, (Output, Output), (Tensor, Tensor),
      Tensor, Output, DataType.Aux[Double], DataType, Shape, Output]

    override type ESTIMATOR = Estimator[
      Tensor, Output, DataType, Shape, (Output, Output),
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
      ((Output, Output), Output)]

    override val estimator: ESTIMATOR = model.estimator
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
    metrics_train: (RegressionMetricsTF, RegressionMetricsTF),
    metrics_test: (RegressionMetricsTF, RegressionMetricsTF),
    summary_dir: Path,
    training_preds: (Tensor, Tensor),
    test_preds: (Tensor, Tensor)) extends ModelRun {

    override type DATA = HeliosDataSet

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
      Tensor, Output, DataType, Shape, Output,
      Tensor, Output, DataType, Shape, Output],
    estimator: Estimator[
      Tensor, Output, DataType, Shape, Output,
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
      (Output, Output)],
    model_prob: SupervisedTrainableModel[
      Tensor, Output, DataType, Shape, Output,
      Tensor, Output, DataType, Shape, Output],
    estimator_prob: Estimator[
      Tensor, Output, DataType, Shape, Output,
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
      (Output, Output)],
    metrics_train: (RegressionMetricsTF, RegressionMetricsTF),
    metrics_test: (RegressionMetricsTF, RegressionMetricsTF),
    summary_dir: Path,
    training_preds: (Tensor, Tensor),
    test_preds: (Tensor, Tensor)) extends ModelRun {

    override type DATA = HeliosDataSet

    override type MODEL = SupervisedTrainableModel[
      Tensor, Output, DataType, Shape, Output,
      Tensor, Output, DataType, Shape, Output]

    override type ESTIMATOR = Estimator[
      Tensor, Output, DataType, Shape, Output,
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape),
      (Output, Output)]
  }


  case class ExperimentType(
    multi_output: Boolean,
    probabilistic_time_lags: Boolean,
    timelag_prediction: String)

  case class ExperimentResult[Results <: ModelRun](
    config: ExperimentType,
    train_data: TLDATA,
    test_data: TLDATA,
    results: Results)

  def plot_and_write_results[Results <: ModelRun](results: ExperimentResult[Results]): Unit = {

    val (
      (data, collated_data),
      (data_test, collated_data_test),
      tf_data,
      (metrics_train, metrics_time_lag_train),
      (metrics_test, metrics_time_lag_test),
      tf_summary_dir,
      train_preds,
      test_preds) = results match {
      case ExperimentResult(
      _,
      (d, cd),
      (dt, cdt),
      JointModelRun(
      (tf_d, _), _, _,
      (m_train, m_time_lag_train),
      (m_test, m_time_lag_test),
      tf_dir,
      tr_preds,
      te_preds)) => (
        (d, cd),
        (dt, cdt),
        tf_d,
        (m_train, m_time_lag_train),
        (m_test, m_time_lag_test),
        tf_dir,
        tr_preds,
        te_preds
      )

      case ExperimentResult(
      _,
      (d, cd),
      (dt, cdt),
      StageWiseModelRun(
      (tf_d, _), _, _, _, _,
      (m_train, m_time_lag_train),
      (m_test, m_time_lag_test),
      tf_dir,
      tr_preds,
      te_preds)) => (
        (d, cd),
        (dt, cdt),
        tf_d,
        (m_train, m_time_lag_train),
        (m_test, m_time_lag_test),
        tf_dir,
        tr_preds,
        te_preds
      )
    }

    val err_time_lag_test = metrics_time_lag_test.preds.subtract(metrics_time_lag_test.targets)

    val mae_lag = err_time_lag_test
      .abs.mean()
      .scalar
      .asInstanceOf[Double]

    val pred_time_lags_test = metrics_time_lag_test.preds

    print("Mean Absolute Error in time lag = ")
    pprint.pprintln(mae_lag)

    plot_histogram(pred_time_lags_test, plot_title = "Predicted Time Lags")
    plot_histogram(err_time_lag_test, plot_title = "Time Lag prediction errors")

    plot_time_series(metrics_test.targets, metrics_test.preds, plot_title = "Test Set Predictions")

    plot_input_output(
      input = collated_data_test.map(_._2._1),
      input_to_scalar = (t: Tensor) => t.square.sum().scalar.asInstanceOf[Float].toDouble,
      targets = metrics_test.targets,
      predictions = metrics_test.preds,
      xlab = "||x(t)||_2",
      ylab = "f(x(t))",
      plot_title = "Input-Output Relationship: Test Data"
    )

    /*
    * Now we compare the predictions of the individual predictions
    * */
    val num_models = collated_data.head._2._2.length

    val time_lag_max = math.min(num_models, collated_data_test.map(_._2._3).max.toInt)

    val selected_indices = (0 to time_lag_max).filter(_ % 2 == 0)

    //Write the model outputs to disk

    write_model_outputs(train_preds, tf_summary_dir, "train")
    write_model_outputs(test_preds,  tf_summary_dir, "test")

    val model_preds = test_preds._1.unstack(num_models, axis = 1)
    val targets     = tf_data.testLabels.unstack(num_models, axis = 1)

    val selected_errors = selected_indices.map(i => model_preds(i)/*.squaredDifference(targets(i)).sqrt*/)

    val probabilities = test_preds._2.unstack(num_models, axis = 1)

    val selected_probabilities = selected_indices.map(probabilities(_))

    /*plot_input_output(
      input = collated_data_test.map(_._2._1),
      input_to_scalar = (t: Tensor) => t.square.sum().scalar.asInstanceOf[Float].toDouble,
      predictions = selected_errors :+ metrics_test.targets,
      xlab = "||x||_2",
      ylab = "Predictor f_i",
      plot_legend = selected_indices.map(i => s"Predictor_$i") :+ "Data",
      plot_title = "Input-Output Relationships: Test Data"
    )*/

    /*selected_indices.foreach(i => {
      plot_input_output(
        input = collated_data_test.map(_._2._1),
        input_to_scalar = (t: Tensor) => t.square.sum().scalar.asInstanceOf[Float].toDouble,
        predictions = Seq(model_preds(i), targets(i)),
        xlab = "||x||_2",
        ylab = "Predictor f_i",
        plot_legend = Seq(s"Predictor_$i", s"Target_$i"),
        plot_title = "Input-Output Relationships: Test Data"
      )
    })*/

    /*plot_input_output(
      input = collated_data_test.map(_._2._1),
      input_to_scalar = (t: Tensor) => t.square.sum().scalar.asInstanceOf[Float].toDouble,
      predictions = selected_probabilities,
      xlab = "||x||_2",
      ylab = "p_i",
      plot_legend = selected_indices.map(i => s"Predictor_$i"),
      plot_title = "Input-Output/Probability Relationships: Test Data"
    )*/


    plot_time_series(
      metrics_train.targets,
      metrics_train.preds, plot_title =
        "Training Set Predictions")

    /*plot_input_output(
      input = collated_data.map(_._2._1),
      input_to_scalar = (t: Tensor) => t.square.sum().scalar.asInstanceOf[Float].toDouble,
      targets = metrics_train.targets,
      predictions = metrics_train.preds,
      xlab = "||x||_2",
      ylab = "f(x(t))",
      plot_title = "Input-Output Relationship: Training Data"
    )*/


    val err_train     = metrics_train.preds.subtract(metrics_train.targets)
    val err_lag_train = metrics_time_lag_train.preds.subtract(metrics_time_lag_train.targets)

    val train_err_scatter = plot_scatter(
      err_train,
      err_lag_train,
      xlab = Some("Error in Velocity"),
      ylab = Some("Error in Time Lag"),
      plot_title = Some("Training Set Errors; Scatter")
    )


    write_data_set((data, collated_data), tf_summary_dir, "train_data")

    //Write errors in target vs errors in time lag for the training data.
    write(
      tf_summary_dir/"train_errors.csv",
      "error_v,error_lag\n"+train_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))


    val train_scatter = plot_scatter(
      metrics_train.preds,
      metrics_time_lag_train.preds,
      xlab = Some("Velocity"),
      ylab = Some("Time Lag"),
      plot_title = Some("Training Set; Scatter"))

    hold()

    val train_actual_scatter = plot_scatter(
      metrics_train.targets,
      metrics_time_lag_train.targets)
    legend(Seq("Predictions", "Actual Data"))
    unhold()


    write_predictions_and_gt(train_scatter, train_actual_scatter, tf_summary_dir, "train")

    val err_test     = metrics_test.preds.subtract(metrics_test.targets)
    val err_lag_test = metrics_time_lag_test.preds.subtract(metrics_time_lag_test.targets)

    val test_err_scatter = plot_scatter(
      err_test, err_lag_test,
      xlab = Some("Error in Velocity"),
      ylab = Some("Error in Time Lag"),
      plot_title = Some("Test Set Errors; Scatter")
    )

    //Write errors in target vs errors in time lag for the test data.
    write(
      tf_summary_dir/"test_errors.csv",
      "error_v,error_lag\n"+test_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))

    write_data_set((data_test, collated_data_test), tf_summary_dir, "test_data")

    val test_scatter = plot_scatter(
      metrics_test.preds,
      metrics_time_lag_test.preds,
      xlab = Some("Velocity"),
      ylab = Some("Time Lag"),
      plot_title = Some("Test Set; Scatter"))

    hold()

    val test_actual_scatter = plot_scatter(
      metrics_test.targets,
      metrics_time_lag_test.targets)

    legend(Seq("Predictions", "Actual Data"))
    unhold()

    write_predictions_and_gt(test_scatter, test_actual_scatter, tf_summary_dir, "test")

    val script1 = pwd/'helios/'scripts/"visualise_tl_results.R"
    val script2 = pwd/'helios/'scripts/"visualise_tl_preds.R"

    try {
      %%('Rscript, script1, tf_summary_dir)
    } catch {
      case e: Exception => e.printStackTrace()
    }

    try {
      %%('Rscript, script2, tf_summary_dir)
    } catch {
      case e: Exception => e.printStackTrace()
    }

  }


  //Runs an experiment given some architecture, loss and training parameters.
  def run_exp(
    dataset: TLDATA,
    architecture: Layer[Output, (Output, Output)],
    loss: Layer[((Output, Output), Output), Output],
    iterations: Int               = 150000,
    optimizer: Optimizer          = tf.train.AdaDelta(0.01),
    miniBatch: Int                = 512,
    sum_dir_prefix: String        = "",
    mo_flag: Boolean              = false,
    prob_timelags: Boolean        = false,
    timelag_pred_strategy: String = "mode",
    summaries_top_dir: Path       = home/'tmp): ExperimentResult[JointModelRun] = {

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
      mo_flag, prob_timelags, timelag_pred_strategy,
      summaries_top_dir
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
    timelag_pred_strategy: String = "mode",
    summaries_top_dir: Path       = home/'tmp): ExperimentResult[JointModelRun] = {

    val (data, collated_data): TLDATA           = dataset._1
    val (data_test, collated_data_test): TLDATA = dataset._2

    val causal_window  = collated_data.head._2._2.length
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

        val tf_summary_dir     = summaries_top_dir/summary_dir_index

        val input              = tf.learn.Input(FLOAT64, Shape(-1, tf_dataset.trainData.shape(1)))

        val trainInput         = tf.learn.Input(FLOAT64, Shape(-1, causal_window))

        val trainingInputLayer = tf.learn.Cast("TrainInput", FLOAT64)

        val summariesDir       = java.nio.file.Paths.get(tf_summary_dir.toString())

        val (model, estimator) = dtflearn.build_tf_model(
          architecture, input, trainInput, trainingInputLayer,
          loss, optimizer, summariesDir,
          dtflearn.rel_loss_change_stop(0.05, iterations))(
          training_data)



        val predictions_training: (Tensor, Tensor) = estimator.infer(() => tf_dataset.trainData)

        val (pred_outputs_train, pred_time_lags_train) = process_predictions(
          predictions_training,
          collated_data.head._2._2.length,
          mo_flag,
          prob_timelags,
          timelag_pred_strategy,
          Some(scalers._2))


        val unscaled_train_labels = scalers._2.i(tf_dataset.trainLabels)

        val actual_outputs_train = (0 until tf_dataset.nTrain).map(n => {
          val time_lag = pred_time_lags_train(n).scalar.asInstanceOf[Double].toInt
          unscaled_train_labels(n, time_lag).scalar.asInstanceOf[Double]
        })

        val metrics_time_lag_train = new RegressionMetricsTF(pred_time_lags_train, train_time_lags)
        metrics_time_lag_train.target_quantity_("Time Lag: Train Data Set")

        val metrics_output_train   = new RegressionMetricsTF(pred_outputs_train, actual_outputs_train)
        metrics_output_train.target_quantity_("Output: Train Data Set")


        val predictions_test: (Tensor, Tensor) = estimator.infer(() => tf_dataset.testData)

        val (pred_outputs_test, pred_time_lags_test) = process_predictions(
          predictions_test,
          causal_window,
          mo_flag,
          prob_timelags,
          timelag_pred_strategy,
          Some(scalers._2))

        val actual_outputs_test = (0 until num_test).map(n => {
          val time_lag = pred_time_lags_test(n).scalar.asInstanceOf[Double].toInt
          tf_dataset.testLabels(n, time_lag).scalar.asInstanceOf[Double]
        })

        val metrics_time_lag_test = new RegressionMetricsTF(pred_time_lags_test, test_time_lags)
        metrics_time_lag_test.target_quantity_("Time Lag: Test Data Set")

        val metrics_output_test   = new RegressionMetricsTF(pred_outputs_test, actual_outputs_test)
        metrics_output_test.target_quantity_("Output: Test Data Set")

        JointModelRun(
          (tf_dataset, scalers),
          model,
          estimator,
          (metrics_output_train, metrics_time_lag_train),
          (metrics_output_test, metrics_time_lag_test),
          tf_summary_dir,
          (scalers._2.i(predictions_training._1), predictions_training._2),
          (scalers._2.i(predictions_test._1), predictions_test._2)
        )

      })

    //The processing pipeline
    val train_and_evaluate =
      data_splits_to_tensors(causal_window) >
        scale_data_v1 >
        model_train_eval

    val results_model_eval = train_and_evaluate(collated_data, collated_data_test)

    val exp_results = ExperimentResult(
      ExperimentType(mo_flag, prob_timelags, timelag_pred_strategy),
      (data, collated_data), (data_test, collated_data_test),
      results_model_eval
    )

    exp_results

  }

  def run_exp3(
    dataset: (TLDATA, TLDATA),
    architecture_i: Layer[Output, Output],
    architecture_ii: Layer[Output, Output],
    loss_i: Layer[(Output, Output), Output],
    loss_ii: Layer[(Output, Output), Output],
    iterations: Int               = 150000,
    optimizer: Optimizer          = tf.train.AdaDelta(0.01),
    miniBatch: Int                = 512,
    sum_dir_prefix: String        = "",
    mo_flag: Boolean              = false,
    prob_timelags: Boolean        = false,
    timelag_pred_strategy: String = "mode",
    summaries_top_dir: Path       = home/'tmp): ExperimentResult[StageWiseModelRun] = {

    val (data, collated_data): TLDATA           = dataset._1
    val (data_test, collated_data_test): TLDATA = dataset._2

    val causal_window  = collated_data.head._2._2.length
    val num_test       = collated_data_test.length

    val model_train_eval = DataPipe(
      (dataTuple: ((HeliosDataSet, (GaussianScalerTF, GaussianScalerTF)), (Tensor, Tensor))) => {

        val ((tf_dataset, scalers), (train_time_lags, test_time_lags)) = dataTuple

        //The first model
        val training_data_i = tf.data.TensorSlicesDataset(tf_dataset.trainData)
          .zip(tf.data.TensorSlicesDataset(tf_dataset.trainLabels)).repeat()
          .shuffle(10)
          .batch(miniBatch)
          .prefetch(10)

        val dt = DateTime.now()

        val summary_dir_index  =
          if(mo_flag) sum_dir_prefix+"_timelag_inference_mo_"+dt.toString("YYYY-MM-dd-HH-mm")
          else sum_dir_prefix+"_timelag_inference_"+dt.toString("YYYY-MM-dd-HH-mm")


        val tf_summary_dir       = summaries_top_dir/summary_dir_index

        val tf_summary_dir_i     = tf_summary_dir/'model_i

        val input_i              = tf.learn.Input(FLOAT64, Shape(-1, tf_dataset.trainData.shape(1)))

        val trainInput_i         = tf.learn.Input(FLOAT64, Shape(-1, causal_window))

        val trainingInputLayer_i = tf.learn.Cast("TrainInput", FLOAT64)

        val summariesDir_i       = java.nio.file.Paths.get(tf_summary_dir_i.toString())

        println("====================================================================")
        println("Model I: Multivariate Prediction of Targets")
        println("====================================================================")


        val (model_i, estimator_i) = dtflearn.build_tf_model(
          architecture_i, input_i, trainInput_i, trainingInputLayer_i,
          loss_i, optimizer, summariesDir_i,
          dtflearn.rel_loss_change_stop(0.05, iterations))(
          training_data_i)


        println("--------------------------------------------------------------------")
        println("Generating multivariate predictions")
        println("--------------------------------------------------------------------")

        //Generate the output signal predictions for train and test sets.
        val predictions_training_i: Tensor = estimator_i.infer(() => tf_dataset.trainData)
        val predictions_test_i: Tensor     = estimator_i.infer(() => tf_dataset.testData)


        val errors_train                   = predictions_training_i.subtract(tf_dataset.trainLabels).square

        //The second model
        val training_data_ii = tf.data.TensorSlicesDataset(tf_dataset.trainData)
          .zip(tf.data.TensorSlicesDataset(errors_train)).repeat()
          .shuffle(10)
          .batch(miniBatch)
          .prefetch(10)



        val tf_summary_dir_ii     = tf_summary_dir/'model_ii

        val input_ii              = tf.learn.Input(FLOAT64, Shape(-1, tf_dataset.trainData.shape(1)))

        val trainInput_ii         = tf.learn.Input(FLOAT64, Shape(-1, causal_window))

        val trainingInputLayer_ii = tf.learn.Cast("TrainInput", FLOAT64)

        val summariesDir_ii       = java.nio.file.Paths.get(tf_summary_dir_ii.toString())

        println("====================================================================")
        println("Model II: Time Lag Model")
        println("====================================================================")


        val (model_ii, estimator_ii) = dtflearn.build_tf_model(
          architecture_ii, input_ii, trainInput_ii, trainingInputLayer_ii,
          loss_ii, optimizer, summariesDir_ii,
          dtflearn.rel_loss_change_stop(0.05, iterations))(
          training_data_ii)


        //Generate the time lag predictions for train and test sets.
        val predictions_training_ii: Tensor = estimator_ii.infer(() => tf_dataset.trainData)
        val predictions_test_ii: Tensor     = estimator_ii.infer(() => tf_dataset.testData)


        println("--------------------------------------------------------------------")
        println("Processing predictions")
        println("--------------------------------------------------------------------")

        val (pred_outputs_train, pred_time_lags_train) = process_predictions(
          (predictions_training_i, predictions_training_ii),
          collated_data.head._2._2.length,
          mo_flag,
          prob_timelags,
          timelag_pred_strategy,
          Some(scalers._2))


        val unscaled_train_labels = scalers._2.i(tf_dataset.trainLabels)

        val actual_outputs_train = (0 until tf_dataset.nTrain).map(n => {
          val time_lag = pred_time_lags_train(n).scalar.asInstanceOf[Double].toInt
          unscaled_train_labels(n, time_lag).scalar.asInstanceOf[Double]
        })

        val metrics_time_lag_train = new RegressionMetricsTF(pred_time_lags_train, train_time_lags)
        metrics_time_lag_train.target_quantity_("Time Lag: Train Data Set")

        val metrics_output_train   = new RegressionMetricsTF(pred_outputs_train, actual_outputs_train)
        metrics_output_train.target_quantity_("Output: Train Data Set")


        val (pred_outputs_test, pred_time_lags_test) = process_predictions(
          (predictions_test_i, predictions_test_ii),
          causal_window,
          mo_flag,
          prob_timelags,
          timelag_pred_strategy,
          Some(scalers._2))

        val actual_outputs_test = (0 until num_test).map(n => {
          val time_lag = pred_time_lags_test(n).scalar.asInstanceOf[Double].toInt
          tf_dataset.testLabels(n, time_lag).scalar.asInstanceOf[Double]
        })

        val metrics_time_lag_test = new RegressionMetricsTF(pred_time_lags_test, test_time_lags)
        metrics_time_lag_test.target_quantity_("Time Lag: Test Data Set")

        val metrics_output_test   = new RegressionMetricsTF(pred_outputs_test, actual_outputs_test)
        metrics_output_test.target_quantity_("Output: Test Data Set")

        StageWiseModelRun(
          (tf_dataset, scalers),
          model_i,
          estimator_i,
          model_ii,
          estimator_ii,
          (metrics_output_train, metrics_time_lag_train),
          (metrics_output_test, metrics_time_lag_test),
          tf_summary_dir,
          (scalers._2.i(predictions_training_i), predictions_training_ii),
          (scalers._2.i(predictions_test_i), predictions_test_ii)
        )

      })

    //The processing pipeline
    val train_and_evaluate =
      data_splits_to_tensors(causal_window) >
        scale_data_v1 >
        model_train_eval

    val results_model_eval = train_and_evaluate(collated_data, collated_data_test)

    val exp_results = ExperimentResult(
      ExperimentType(mo_flag, prob_timelags, timelag_pred_strategy),
      (data, collated_data), (data_test, collated_data_test),
      results_model_eval
    )

    exp_results
  }

  def run_exp_hyp(
    dataset: (TLDATA, TLDATA),
    arch: Layer[Output, (Output, Output)],
    hyper_params: List[String],
    loss_func_generator: dtflearn.tunable_tf_model.HyperParams => Layer[((Output, Output), Output), Output],
    fitness_func: DataPipe2[(Tensor, Tensor), Tensor, Double],
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]],
    iterations: Int               = 150000,
    iterations_tuning: Int        = 20000,
    num_samples: Int              = 20,
    optimizer: Optimizer          = tf.train.AdaDelta(0.01),
    miniBatch: Int                = 512,
    sum_dir_prefix: String        = "",
    mo_flag: Boolean              = false,
    prob_timelags: Boolean        = false,
    timelag_pred_strategy: String = "mode",
    summaries_top_dir: Path       = home/'tmp): ExperimentResult[TunedModelRun] = {

    val (data, collated_data): TLDATA           = dataset._1
    val (data_test, collated_data_test): TLDATA = dataset._2

    val data_size = collated_data.toStream.length

    val causal_window = collated_data.head._2._2.length
    val input_shape   = collated_data.head._2._1.shape
    val num_test      = collated_data_test.length

    type SC_DATA = (helios.data.TF_DATA, (GaussianScalerTF, GaussianScalerTF))

    val model_train_eval = DataPipe((data_and_scales: (SC_DATA, (Tensor, Tensor))) => {

      val ((tfdata, scalers), (train_time_lags, test_time_lags)) = data_and_scales

      val dt = DateTime.now()

      val summary_dir_index  =
        if(mo_flag) sum_dir_prefix+"_timelag_mo_"+dt.toString("YYYY-MM-dd-HH-mm")
        else sum_dir_prefix+"_timelag_"+dt.toString("YYYY-MM-dd-HH-mm")

      val tf_summary_dir     = summaries_top_dir/summary_dir_index

      val train_config_tuning = dtflearn.model.trainConfig(
        tf_summary_dir, optimizer,
        dtflearn.rel_loss_change_stop(0.005, iterations_tuning)
      )

      val tf_data_ops = dtflearn.model.data_ops(10, miniBatch, 10, data_size/5)

      val stackOperation = DataPipe[Iterable[Tensor], Tensor](bat =>
        tfi.stack(bat.toSeq, axis = 0)
      )

      val tunableTFModel: TunableTFModel[
        Tensor, Output, DataType.Aux[Float], DataType, Shape, (Output, Output), (Tensor, Tensor),
        Tensor, Output, DataType.Aux[Double], DataType, Shape, Output] =
        dtflearn.tunable_tf_model(
          loss_func_generator, hyper_params,
          tfdata.training_dataset,
          fitness_func,
          arch,
          (FLOAT32, input_shape),
          (FLOAT64, Shape(causal_window)),
          tf.learn.Cast("TrainInput", FLOAT64),
          train_config_tuning,
          data_split_func = Some(DataPipe[(Tensor, Tensor), Boolean](_ => scala.util.Random.nextGaussian() <= 0.7)),
          data_processing = tf_data_ops,
          inMemory = false,
          concatOp = Some(stackOperation)
      )

      val gs = new GridSearch[tunableTFModel.type](tunableTFModel)

      gs.setPrior(hyper_prior)

      gs.setNumSamples(num_samples)


      println("--------------------------------------------------------------------")
      println("Initiating model tuning")
      println("--------------------------------------------------------------------")

      val (_, config) = gs.optimize(hyper_prior.mapValues(_.draw))

      println("--------------------------------------------------------------------")
      println("Model tuning complete")
      println("Chosen configuration:")
      pprint.pprintln(config)
      println("--------------------------------------------------------------------")

      println("Training final model based on chosen configuration")

      val model_function = dtflearn.tunable_tf_model.ModelFunction.from_loss_generator[
        Tensor, Output, DataType.Aux[Float], DataType, Shape, (Output, Output), (Tensor, Tensor),
        Tensor, Output, DataType.Aux[Double], DataType, Shape, Output
        ](
        loss_func_gen, arch, (FLOAT32, input_shape),
        (FLOAT64, Shape(causal_window)),
        tf.learn.Cast("TrainInput", FLOAT64),
        train_config_tuning.copy(stopCriteria =  dtflearn.rel_loss_change_stop(0.005, iterations)),
        tf_data_ops, inMemory = false,
        concatOpI = Some(stackOperation),
        concatOpT = Some(stackOperation)
      )

      val best_model = model_function(config)(tfdata.training_dataset)

      val extract_features = (p: (Tensor, Tensor)) => p._1
      val extract_preds    = (p: (Tensor, (Tensor, Tensor))) => p._2

      val test_predictions = collect_predictions(
        best_model.infer_coll(tfdata.test_dataset.map(extract_features)).map(extract_preds)
      )

      val train_predictions = collect_predictions(
        best_model.infer_coll(tfdata.training_dataset.map(extract_features)).map(extract_preds)
      )

      val (pred_outputs_train, pred_time_lags_train) = process_predictions(
        train_predictions,
        collated_data.head._2._2.length,
        mo_flag,
        prob_timelags,
        timelag_pred_strategy,
        Some(scalers._2))


      val unscaled_train_labels =
        tfdata.training_dataset.map((p: (Tensor, Tensor)) => p._2).map(scalers._2.i).data.toSeq

      val actual_outputs_train = (0 until tfdata.training_dataset.size).map(n => {
        val time_lag = pred_time_lags_train(n).scalar.asInstanceOf[Double].toInt
        unscaled_train_labels(n)(time_lag).scalar.asInstanceOf[Double]
      })

      val metrics_time_lag_train = new RegressionMetricsTF(pred_time_lags_train, train_time_lags)
      metrics_time_lag_train.target_quantity_("Time Lag: Train Data Set")

      val metrics_output_train   = new RegressionMetricsTF(pred_outputs_train, actual_outputs_train)
      metrics_output_train.target_quantity_("Output: Train Data Set")


      val (pred_outputs_test, pred_time_lags_test) = process_predictions(
        test_predictions,
        causal_window,
        mo_flag,
        prob_timelags,
        timelag_pred_strategy,
        Some(scalers._2))

      val unscaled_test_labels = tfdata.test_dataset.map((p: (Tensor, Tensor)) => p._2).data.toSeq

      val actual_outputs_test = (0 until num_test).map(n => {
        val time_lag = pred_time_lags_test(n).scalar.asInstanceOf[Double].toInt
        unscaled_test_labels(n)(time_lag).scalar.asInstanceOf[Double]
      })

      val metrics_time_lag_test = new RegressionMetricsTF(pred_time_lags_test, test_time_lags)
      metrics_time_lag_test.target_quantity_("Time Lag: Test Data Set")

      val metrics_output_test   = new RegressionMetricsTF(pred_outputs_test, actual_outputs_test)
      metrics_output_test.target_quantity_("Output: Test Data Set")



      TunedModelRun(
        data_and_scales._1,
        best_model,
        (metrics_output_train, metrics_time_lag_train),
        (metrics_output_test, metrics_time_lag_test),
        tf_summary_dir,
        (scalers._2.i(train_predictions._1), train_predictions._2),
        (scalers._2.i(test_predictions._1), test_predictions._2)
      )
    })



    //The processing pipeline
    val train_and_evaluate =
      data_splits_to_dataset(causal_window) >
        scale_data_v2 >
        model_train_eval

    val results_model_eval = train_and_evaluate(collated_data, collated_data_test)

    val exp_results = ExperimentResult(
      ExperimentType(mo_flag, prob_timelags, timelag_pred_strategy),
      (data, collated_data), (data_test, collated_data_test),
      results_model_eval
    )

    exp_results

  }


}
