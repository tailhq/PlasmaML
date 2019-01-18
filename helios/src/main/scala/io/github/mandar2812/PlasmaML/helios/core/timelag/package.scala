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
import _root_.io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GridSearch, CMAES}
import _root_.io.github.mandar2812.dynaml.models.{TFModel, TunableTFModel}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data.HeliosDataSet
import org.platanios.tensorflow.api.learn.estimators.Estimator
import org.platanios.tensorflow.api.learn.{INFERENCE, Mode, StopCriteria, SupervisedTrainableModel}
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag.utils._
import breeze.stats.distributions.ContinuousDistr

import org.json4s._
import org.json4s.jackson.Serialization.{read => read_json, write => write_json}

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

  def plot_and_write_results[Results <: ModelRun](
    results: ExperimentResult[Results],
    browser_plots: Boolean = true): Unit = {

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


    val err_train     = metrics_train.preds.subtract(metrics_train.targets)
    val err_lag_train = metrics_time_lag_train.preds.subtract(metrics_time_lag_train.targets)

    val train_scatter =
      dtfutils.toDoubleSeq(metrics_train.preds).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_train.preds)
      ).toSeq

    val train_actual_scatter =
      dtfutils.toDoubleSeq(metrics_train.targets).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_train.targets)
      ).toSeq


    val train_err_scatter =
      dtfutils.toDoubleSeq(err_train).zip(
        dtfutils.toDoubleSeq(err_lag_train)
      ).toSeq

    val err_test     = metrics_test.preds.subtract(metrics_test.targets)
    val err_lag_test = metrics_time_lag_test.preds.subtract(metrics_time_lag_test.targets)

    val test_err_scatter = dtfutils.toDoubleSeq(err_test).zip(dtfutils.toDoubleSeq(err_lag_test)).toSeq


    val test_scatter =
      dtfutils.toDoubleSeq(metrics_test.preds).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_test.preds)
      ).toSeq

    val test_actual_scatter =
      dtfutils.toDoubleSeq(metrics_test.targets).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_test.targets)
      ).toSeq


    if(browser_plots) {

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

      plot_time_series(
        metrics_train.targets,
        metrics_train.preds, plot_title =
          "Training Set Predictions")


      plot_scatter(
        train_err_scatter,
        xlab = Some("Error in Velocity"),
        ylab = Some("Error in Time Lag"),
        plot_title = Some("Training Set Errors; Scatter")
      )

      plot_scatter(
        train_scatter,
        xlab = Some("Velocity"),
        ylab = Some("Time Lag"),
        plot_title = Some("Training Set; Scatter"))

      hold()

      plot_scatter(train_actual_scatter)
      legend(Seq("Predictions", "Actual Data"))

      unhold()

      plot_scatter(
        test_err_scatter,
        xlab = Some("Error in Velocity"),
        ylab = Some("Error in Time Lag"),
        plot_title = Some("Test Set Errors; Scatter")
      )

      plot_scatter(
        test_scatter,
        xlab = Some("Velocity"),
        ylab = Some("Time Lag"),
        plot_title = Some("Test Set; Scatter"))

      hold()

      plot_scatter(test_actual_scatter)

      legend(Seq("Predictions", "Actual Data"))

      unhold()

    }

    write_data_set((data, collated_data), tf_summary_dir, "train_data")
    write_data_set((data_test, collated_data_test), tf_summary_dir, "test_data")

    write_predictions_and_gt(train_scatter, train_actual_scatter, tf_summary_dir, "train")
    write_predictions_and_gt(test_scatter, test_actual_scatter, tf_summary_dir, "test")

    //Write the model outputs to disk
    write_model_outputs(train_preds, tf_summary_dir, "train")
    write_model_outputs(test_preds,  tf_summary_dir, "test")

    //Write errors in target vs errors in time lag for the training data.
    write(
      tf_summary_dir/"train_errors.csv",
      "error_v,error_lag\n"+train_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))

    //Write errors in target vs errors in time lag for the test data.
    write(
      tf_summary_dir/"test_errors.csv",
      "error_v,error_lag\n"+test_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))


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


  def plot_and_write_results_tuned(
    results: ExperimentResult[TunedModelRun],
    browser_plots: Boolean = true): Unit = {

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
          TunedModelRun(
          (tf_d, _), _, 
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


    val err_train     = metrics_train.preds.subtract(metrics_train.targets)
    val err_lag_train = metrics_time_lag_train.preds.subtract(metrics_time_lag_train.targets)

    val train_scatter =
      dtfutils.toDoubleSeq(metrics_train.preds).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_train.preds)
      ).toSeq

    val train_actual_scatter =
      dtfutils.toDoubleSeq(metrics_train.targets).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_train.targets)
      ).toSeq


    val train_err_scatter =
      dtfutils.toDoubleSeq(err_train).zip(
        dtfutils.toDoubleSeq(err_lag_train)
      ).toSeq

    val err_test     = metrics_test.preds.subtract(metrics_test.targets)
    val err_lag_test = metrics_time_lag_test.preds.subtract(metrics_time_lag_test.targets)

    val test_err_scatter = dtfutils.toDoubleSeq(err_test).zip(dtfutils.toDoubleSeq(err_lag_test)).toSeq


    val test_scatter =
      dtfutils.toDoubleSeq(metrics_test.preds).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_test.preds)
      ).toSeq

    val test_actual_scatter =
      dtfutils.toDoubleSeq(metrics_test.targets).zip(
        dtfutils.toDoubleSeq(metrics_time_lag_test.targets)
      ).toSeq


    if(browser_plots) {

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

      plot_time_series(
        metrics_train.targets,
        metrics_train.preds, plot_title =
          "Training Set Predictions")


      plot_scatter(
        train_err_scatter,
        xlab = Some("Error in Velocity"),
        ylab = Some("Error in Time Lag"),
        plot_title = Some("Training Set Errors; Scatter")
      )

      plot_scatter(
        train_scatter,
        xlab = Some("Velocity"),
        ylab = Some("Time Lag"),
        plot_title = Some("Training Set; Scatter"))

      hold()

      plot_scatter(train_actual_scatter)
      legend(Seq("Predictions", "Actual Data"))

      unhold()

      plot_scatter(
        test_err_scatter,
        xlab = Some("Error in Velocity"),
        ylab = Some("Error in Time Lag"),
        plot_title = Some("Test Set Errors; Scatter")
      )

      plot_scatter(
        test_scatter,
        xlab = Some("Velocity"),
        ylab = Some("Time Lag"),
        plot_title = Some("Test Set; Scatter"))

      hold()

      plot_scatter(test_actual_scatter)

      legend(Seq("Predictions", "Actual Data"))

      unhold()

    }

    write_data_set((data, collated_data), tf_summary_dir, "train_data")
    write_data_set((data_test, collated_data_test), tf_summary_dir, "test_data")

    write_predictions_and_gt(train_scatter, train_actual_scatter, tf_summary_dir, "train")
    write_predictions_and_gt(test_scatter, test_actual_scatter, tf_summary_dir, "test")

    //Write the model outputs to disk
    write_model_outputs(train_preds, tf_summary_dir, "train")
    write_model_outputs(test_preds,  tf_summary_dir, "test")

    //Write errors in target vs errors in time lag for the training data.
    write(
      tf_summary_dir/"train_errors.csv",
      "error_v,error_lag\n"+train_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))

    //Write errors in target vs errors in time lag for the test data.
    write(
      tf_summary_dir/"test_errors.csv",
      "error_v,error_lag\n"+test_err_scatter.map(c => c._1.toString + "," + c._2.toString).mkString("\n"))


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

  def close_plots(): Unit = stopServer

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


    run_exp_joint(
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


  /**
    * <h4>Causal Time Lag: Joint Inference</h4>
    *
    * Runs a model train-evaluation experiment. The model
    * is trained to predict output labels for the entire
    * causal window and a prediction for the causal time lag link
    * between multi-dimensional input time series x(t) and a
    * one dimensional output series y(t).
    *
    * @param dataset The training and test data tuple, each one of
    *                type [[TLDATA]].
    * @param architecture The neural architecture making the predictions.
    * @param loss The loss function used to train the models, see the [[helios.learn]]
    *             package for the kind of loss functions implemented.
    *
    * @param iterations The max number of training iterations to run.
    * @param optimizer The optimization algorithm to use for training
    *                  model parameters.
    * @param miniBatch Size of one data batch, used for gradient based learning.
    * @param sum_dir_prefix The string prefix given to the model's summary/results
    *                       directory.
    * @param mo_flag If the model making one prediction for the entire causal window, then
    *                set to false. If the model is making one prediction for each output
    *                in the causal window, set to true.
    * @param prob_timelags If the model is making probabilisitc prediction of the causal
    *                      time lag, then set to true.
    * @param timelag_pred_strategy In case of probabilistic time lag prediction, how
    *                              should the target be chosen. Defaults to "mode", meaning
    *                              the most likely output in the prediction is given
    *                              as the target prediction.
    *
    * @param summaries_top_dir The top level directory under which the model summary directory
    *                          will be created, defaults to ~/tmp
    *
    * @return An [[ExperimentResult]] object which contains
    *         the evaluation results of type [[JointModelRun]]
    *
    * */
  def run_exp_joint(
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
    summaries_top_dir: Path       = home/'tmp,
    epochFlag: Boolean            = false): ExperimentResult[JointModelRun] = {

    val (data, collated_data): TLDATA           = dataset._1
    val (data_test, collated_data_test): TLDATA = dataset._2

    val data_size = collated_data.toSeq.length

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

        val stopCondition      = get_stop_condition(iterations, 0.05, epochFlag, data_size, miniBatch)

        val (model, estimator) = dtflearn.build_tf_model(
          architecture, input, trainInput, trainingInputLayer,
          loss, optimizer, summariesDir,
          stopCondition)(
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

  /**
    * <h4>Causal Time Lag: Stage Wise Inference</h4>
    *
    * Runs a model train-evaluation experiment. The model
    * is trained in two stages (each with a different model and architecture).
    *
    * <ol>
    *   <li>Train to predict output labels</li>
    *   <li>Train model for predicting the causal time lag link</li>
    * </ol>
    *
    * @param dataset The training and test data tuple, each one of
    *                type [[TLDATA]].
    * @param architecture_i The neural architecture making the output predictions.
    * @param architecture_ii The neural architecture making the time lag predictions.
    * @param loss_i The loss function used to train the model I,  see [[helios.learn.cdt_i]]
    * @param loss_ii The loss function used to train the model II, see [[helios.learn.cdt_ii]]
    *
    * @param iterations The max number of training iterations to run.
    * @param optimizer The optimization algorithm to use for training
    *                  model parameters.
    * @param miniBatch Size of one data batch, used for gradient based learning.
    * @param sum_dir_prefix The string prefix given to the model's summary/results
    *                       directory.
    * @param mo_flag If the model making one prediction for the entire causal window, then
    *                set to false. If the model is making one prediction for each output
    *                in the causal window, set to true.
    * @param prob_timelags If the model is making probabilisitc prediction of the causal
    *                      time lag, then set to true.
    * @param timelag_pred_strategy In case of probabilistic time lag prediction, how
    *                              should the target be chosen. Defaults to "mode", meaning
    *                              the most likely output in the prediction is given
    *                              as the target prediction.
    *
    * @param summaries_top_dir The top level directory under which the model summary directory
    *                          will be created, defaults to ~/tmp
    *
    * @return An [[ExperimentResult]] object which contains
    *         the evaluation results of type [[StageWiseModelRun]]
    *
    * */
  def run_exp_stage_wise(
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
    summaries_top_dir: Path       = home/'tmp,
    epochFlag: Boolean = false): ExperimentResult[StageWiseModelRun] = {

    val (data, collated_data): TLDATA           = dataset._1
    val (data_test, collated_data_test): TLDATA = dataset._2

    val data_size = collated_data.toSeq.length

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
          get_stop_condition(iterations, 0.05, epochFlag, data_size, miniBatch))(
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
          get_stop_condition(iterations, 0.05, epochFlag, data_size, miniBatch))(
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

  /**
    * <h4>Causal Time Lag: Joint Inference</h4>
    * <h5>Hyper-parameter Tuning</h5>
    *
    * Runs a model train-tune-evaluation experiment.
    *
    * The model is trained to predict output labels for the entire
    * causal window and a prediction for the causal time lag link
    * between multi-dimensional input time series x(t) and a
    * one dimensional output series y(t).
    *
    * The hyper-parameters of the loss function are determined using
    * hyper-parameter optimization algorithms such as [[GridSearch]] and
    * [[CoupledSimulatedAnnealing]].
    *
    * @param dataset The training and test data tuple, each one of
    *                type [[TLDATA]].
    * @param architecture The neural architecture making the predictions.
    * @param loss_func_generator A function which takes the hyper-parameters
    *                            [[dtflearn.tunable_tf_model.HyperParams]] and returns
    *                            an instantiated loss function.
    * @param iterations The max number of training iterations to run.
    * @param iterations_tuning The max number of iterations of training to run for each model instance
    *                          during the tuning process.
    * @param optimizer The optimization algorithm to use for training
    *                  model parameters.
    * @param miniBatch Size of one data batch, used for gradient based learning.
    * @param sum_dir_prefix The string prefix given to the model's summary/results
    *                       directory.
    * @param mo_flag If the model making one prediction for the entire causal window, then
    *                set to false. If the model is making one prediction for each output
    *                in the causal window, set to true.
    * @param prob_timelags If the model is making probabilisitc prediction of the causal
    *                      time lag, then set to true.
    * @param timelag_pred_strategy In case of probabilistic time lag prediction, how
    *                              should the target be chosen. Defaults to "mode", meaning
    *                              the most likely output in the prediction is given
    *                              as the target prediction.
    * @param summaries_top_dir The top level directory under which the model summary directory
    *                          will be created, defaults to ~/tmp
    * @param num_samples The number of hyper-parameter samples to generate from
    *                    the hyper-parameter prior distribution.
    * @param hyper_optimizer The hyper parameter optimization algorithm to use,
    *                        either "gs" (grid search) or "csa" (coupled simulated annealing).
    * @param hyp_opt_iterations If `hyper_optimizer` is set to "csa", then this parameter is used
    *                           for setting the number of csa iterations.
    *
    * @return An [[ExperimentResult]] object which contains
    *         the evaluation results of type [[TunedModelRun]]
    *
    * */
  def run_exp_hyp(
    dataset: (TLDATA, TLDATA),
    architecture: Layer[Output, (Output, Output)],
    hyper_params: List[String],
    loss_func_generator: dtflearn.tunable_tf_model.HyperParams => Layer[((Output, Output), Output), Output],
    fitness_func: DataPipe2[(Tensor, Tensor), Tensor, Double],
    hyper_prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]],
    iterations: Int                 = 150000,
    iterations_tuning: Int          = 20000,
    optimizer: Optimizer            = tf.train.AdaDelta(0.01),
    miniBatch: Int                  = 512,
    sum_dir_prefix: String          = "",
    mo_flag: Boolean                = false,
    prob_timelags: Boolean          = false,
    timelag_pred_strategy: String   = "mode",
    summaries_top_dir: Path         = home/'tmp,
    num_samples: Int                = 20,
    hyper_optimizer: String         = "gs",
    hyp_opt_iterations: Option[Int] = Some(5),
    hyp_mapping: Option[Map[String, Encoder[Double, Double]]] = None,
    epochFlag: Boolean              = false): ExperimentResult[TunedModelRun] = {

    val (data, collated_data): TLDATA           = dataset._1
    val (data_test, collated_data_test): TLDATA = dataset._2

    val data_size = collated_data.toSeq.length

    val data_size_test = collated_data_test.toSeq.length

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

      val stop_condition_tuning = get_stop_condition(iterations_tuning, 0.01, epochFlag, data_size, miniBatch)

      val stop_condition_test   = get_stop_condition(iterations, 0.01, epochFlag, data_size, miniBatch)

      val train_config_tuning =
        dtflearn.tunable_tf_model.ModelFunction.hyper_params_to_dir >>
          DataPipe((p: Path) => dtflearn.model.trainConfig(
            p, optimizer,
            stop_condition_tuning,
            Some(get_train_hooks(p, iterations_tuning, epochFlag, data_size, miniBatch))
          ))

      val train_config_test = DataPipe[dtflearn.tunable_tf_model.HyperParams, dtflearn.model.Config](_ =>
        dtflearn.model.trainConfig(
          summaryDir = tf_summary_dir,
          stopCriteria = stop_condition_test,
          trainHooks = Some(get_train_hooks(tf_summary_dir, iterations, epochFlag, data_size, miniBatch)))
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
          architecture,
          (FLOAT32, input_shape),
          (FLOAT64, Shape(causal_window)),
          tf.learn.Cast("TrainInput", FLOAT64),
          train_config_tuning(tf_summary_dir),
          data_split_func = Some(DataPipe[(Tensor, Tensor), Boolean](_ => scala.util.Random.nextGaussian() <= 0.7)),
          data_processing = tf_data_ops,
          inMemory = false,
          concatOpI = Some(stackOperation),
          concatOpT = Some(stackOperation)
      )

      val gs = hyper_optimizer match {
        case "csa" =>
          new CoupledSimulatedAnnealing[tunableTFModel.type](
            tunableTFModel, hyp_mapping).setMaxIterations(
            hyp_opt_iterations.getOrElse(5)
          )

        case "gs"  => new GridSearch[tunableTFModel.type](tunableTFModel)


        case "cma" => new CMAES(
          tunableTFModel,
          hyper_params,
          0.8,
          hyp_mapping
        ).setMaxIterations(hyp_opt_iterations.getOrElse(5))

        case _     => new GridSearch[tunableTFModel.type](tunableTFModel)
      }

      gs.setPrior(hyper_prior)

      gs.setNumSamples(num_samples)

      println("--------------------------------------------------------------------")
      println("Initiating model tuning")
      println("--------------------------------------------------------------------")

      val (_, config) = gs.optimize(hyper_prior.mapValues(_.draw))

      println("--------------------------------------------------------------------")
      println("\nModel tuning complete")
      println("Chosen configuration:")
      pprint.pprintln(config)
      println("--------------------------------------------------------------------")

      println("Training final model based on chosen configuration")

      write(
        tf_summary_dir/"state.csv",
        config.keys.mkString(start = "", sep = ",", end = "\n") +
          config.values.mkString(start = "", sep = ",", end = "")
      )

      val model_function = dtflearn.tunable_tf_model.ModelFunction.from_loss_generator[
        Tensor, Output, DataType.Aux[Float], DataType, Shape, (Output, Output), (Tensor, Tensor),
        Tensor, Output, DataType.Aux[Double], DataType, Shape, Output
        ](
        loss_func_generator, architecture, (FLOAT32, input_shape),
        (FLOAT64, Shape(causal_window)),
        tf.learn.Cast("TrainInput", FLOAT64),
        train_config_test,
        tf_data_ops, inMemory = false,
        concatOpI = Some(stackOperation),
        concatOpT = Some(stackOperation)
      )

      val best_model = model_function(config)(tfdata.training_dataset)

      best_model.train()

      val extract_features = (p: (Tensor, Tensor)) => p._1

      val model_predictions_test = best_model.infer_coll(tfdata.test_dataset.map(extract_features))
      val model_predictions_train = best_model.infer_coll(tfdata.training_dataset.map(extract_features))


      val test_predictions = model_predictions_test match {
        case Left(tensor) => tensor
        case Right(collection) => collect_predictions(collection)
      }
      

      val train_predictions = model_predictions_train match {
        case Left(tensor) => tensor
        case Right(collection) => collect_predictions(collection)
      }

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
