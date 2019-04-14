package io.github.mandar2812.PlasmaML.helios.core

import ammonite.ops._
import io.github.mandar2812.dynaml.utils.annotation.Experimental
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.models._
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.evaluation.Performance
import io.github.mandar2812.dynaml.pipes.{
  DataPipe,
  DataPipe2,
  MetaPipe,
  Encoder
}
import io.github.mandar2812.dynaml.tensorflow.data.{DataSet, TFDataSet}

import org.json4s._
import org.json4s.jackson.Serialization.{read => read_json, write => write_json}

import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, TF}
import org.platanios.tensorflow.api._

/**
  * <h3>Probabilistic Dynamic Time Lag Model</h3>
  *
  * @param time_window The size of the time window in steps.
  * @param modelFunction Generates a tensorflow model instance
  *                      from hyper-parameters.
  * @param model_config_func Generates model training configuration
  *                          from hyper-parameters.
  * @param hyp_params A collection of hyper-parameters.
  * @param persistent_hyp_params The subset of the hyper-parameters which
  *                              are not updated.
  * @param params_to_mutable_params A one-to-one invertible mapping between
  *                                 the loss function parameters to the
  *                                 cannonical parameters "alpha" and "sigma_sq".
  * @param training_data The training data collection.
  * @param tf_data_handle_ops An instance of [[dtflearn.model.Ops]], describes
  *                           how the data patterns should be loaded into a
  *                           Tensorflow dataset handle.
  * @param fitness_to_scalar A function which processes all the computed metrics
  *                          and returns a single fitness score.
  * @param validation_data An optional validation data collection.
  *
  * @param data_split_func An optional data pipeline which divides the
  *                        training collection into a train and validation split.
  *
  * */
class PDTModel[Pattern, In, IT, ID, IS, Loss: TF: IsFloatOrDouble](
  val time_window: Int,
  override val modelFunction: TunableTFModel.ModelFunc[
    In,
    Output[Double],
    (Output[Double], Output[Double]),
    Loss,
    IT,
    ID,
    IS,
    Tensor[Double],
    FLOAT64,
    Shape,
    (Tensor[Double], Tensor[Double]),
    (FLOAT64, FLOAT64),
    (Shape, Shape)
  ],
  val model_config_func: dtflearn.tunable_tf_model.ModelConfigFunction[
    In,
    Output[Double]
  ],
  override val hyp_params: Seq[String],
  val persistent_hyp_params: Seq[String],
  val params_to_mutable_params: Encoder[
    dtflearn.tunable_tf_model.HyperParams,
    dtflearn.tunable_tf_model.HyperParams
  ],
  override protected val training_data: DataSet[Pattern],
  override val tf_data_handle_ops: dtflearn.model.TFDataHandleOps[
    Pattern,
    IT,
    Tensor[Double],
    (Tensor[Double], Tensor[Double]),
    In,
    Output[Double]
  ],
  override val fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
    DataPipe[Seq[Tensor[Float]], Double](m =>
        m.map(_.scalar.toDouble).sum / m.length),
  override protected val validation_data: Option[DataSet[Pattern]] = None,
  override protected val data_split_func: Option[DataPipe[Pattern, Boolean]] =
    None)
    extends TunableTFModel[Pattern, In, Output[Double], (Output[Double], Output[Double]), Loss, IT, ID, IS, Tensor[
      Double
    ], FLOAT64, Shape, (Tensor[Double], Tensor[Double]), (FLOAT64, FLOAT64), (Shape, Shape)](
      modelFunction,
      model_config_func,
      hyp_params,
      training_data,
      tf_data_handle_ops,
      Seq(PDTModel.s0, PDTModel.c1, PDTModel.c2),
      fitness_to_scalar,
      validation_data,
      data_split_func
    ) {

  val mutable_params_to_metric_functions: DataPipe[
    dtflearn.tunable_tf_model.HyperParams,
    Seq[
      DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]]
    ]
  ] =
    DataPipe(
      (c: Map[String, Double]) =>
        Seq(
          PDTModel.s0,
          PDTModel.c1(c("alpha"), c("sigma_sq"), time_window),
          PDTModel.c2(c("alpha"), c("sigma_sq"), time_window)
        )
    )

  val params_to_metric_funcs: DataPipe[
    dtflearn.tunable_tf_model.HyperParams,
    Seq[
      DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]]
    ]
  ] = params_to_mutable_params > mutable_params_to_metric_functions

  val metrics_to_mutable_params
    : DataPipe[Seq[Tensor[Float]], dtflearn.tunable_tf_model.HyperParams] =
    DataPipe((s: Seq[Tensor[Float]]) => {
      val s0 = s(0).scalar.toDouble
      val c1 = s(1).scalar.toDouble / s0

      Map(
        "alpha"    -> time_window * (1d - c1) / (c1 * (time_window - 1)),
        "sigma_sq" -> s0 * (time_window - c1) / (time_window - 1)
      )
    })

  private def update(
    p: Map[String, Double],
    h: Map[String, Double]
  ): Map[String, Double] = {

    //Train and evaluate the model on the given hyper-parameters
    val train_config = modelConfigFunc(p)

    val stability_metrics = params_to_metric_funcs(h)
      .zip(PDTModel.stability_quantities)
      .map(fitness_function => {
        Performance[((Output[Double], Output[Double]), (In, Output[Double]))](
          fitness_function._2,
          DataPipe[
            ((Output[Double], Output[Double]), (In, Output[Double])),
            Output[Float]
          ](
            c => fitness_function._1(c._1, c._2._2)
          )
        )
      })

    val updated_params = try {
      val model = train_model(
        p ++ h,
        Some(train_config),
        evaluation_metrics =
          Some(PDTModel.stability_quantities.zip(params_to_metric_funcs(h)))
        //stepTrigger
      )

      println("Updating PDT stability metrics.")
      val metrics = model.evaluate(
        train_data_tf,
        train_split.size,
        stability_metrics,
        train_config.data_processing.copy(shuffleBuffer = 0, repeat = 0),
        true,
        null
      )
      (metrics_to_mutable_params > params_to_mutable_params.i)(metrics)
    } catch {
      case e: java.lang.IllegalStateException =>
        h
      case e: Throwable =>
        e.printStackTrace()
        h
    }

    println("Updated Parameters: ")
    pprint.pprintln(p ++ updated_params)
    updated_params
  }

  def solve(pdt_iterations: Int)(hyper_params: TunableTFModel.HyperParams) = {

    val (p, t) =
      hyper_params.toSeq.partition(kv => persistent_hyp_params.contains(kv._1))

    (1 to pdt_iterations).foldLeft(t.toMap)((s, _) => update(p.toMap, s))
  }

  override def energy(
    hyper_params: TunableTFModel.HyperParams,
    options: Map[String, String]
  ): Double = {

    val p = hyper_params.filterKeys(persistent_hyp_params.contains _)

    //Train and evaluate the model on the given hyper-parameters

    //Start by loading the model configuration,
    //which depends only on the `persistent`
    //hyper-parameters.
    val train_config = modelConfigFunc(p.toMap)

    //The number of times the mutable hyper-parameters
    //will be updated.
    val loop_count = options.getOrElse("loops", "2").toInt

    //Run the hyper-parameter refinement procedure.
    val final_config: Map[String, Double] = solve(loop_count)(hyper_params)

    //Now compute the model fitness score.
    val (fitness, comment) = try {

      val stability_metrics = params_to_metric_funcs(final_config)
        .zip(PDTModel.stability_quantities)
        .map(fitness_function => {
          Performance[((Output[Double], Output[Double]), (In, Output[Double]))](
            fitness_function._2,
            DataPipe[
              ((Output[Double], Output[Double]), (In, Output[Double])),
              Output[Float]
            ](
              c => fitness_function._1(c._1, c._2._2)
            )
          )
        })

      val model = train_model(
        p.toMap ++ final_config,
        Some(train_config),
        evaluation_metrics = Some(
          PDTModel.stability_quantities
            .zip(params_to_metric_funcs(final_config))
        ),
        options.get("evalTrigger").map(_.toInt)
      )

      println("Computing Energy.")
      val e = fitness_to_scalar(
        model.evaluate(
          validation_data_tf,
          validation_split.size,
          stability_metrics,
          train_config.data_processing.copy(shuffleBuffer = 0, repeat = 0),
          true,
          null
        )
      )

      (e, None)
    } catch {
      case e: java.lang.IllegalStateException =>
        (Double.PositiveInfinity, Some(e.getMessage))
      case e: Throwable =>
        e.printStackTrace()
        (Double.PositiveInfinity, Some(e.getMessage))
    }

    //Append the model fitness to the hyper-parameter configuration
    val hyp_config_json = write_json(
      p.toMap ++ final_config ++ Map(
        "energy"  -> fitness,
        "comment" -> comment.getOrElse("")
      )
    )

    //Write the configuration along with its fitness into the model
    //instance's summary directory
    write.append(train_config.summaryDir / "state.json", hyp_config_json + "\n")

    //Return the model fitness.
    fitness
  }

}

object PDTModel {

  final val mutable_params: Seq[String] = Seq("alpha", "sigma_sq")

  val stability_quantities = Seq("s0", "c1", "c2")

  val s0 =
    DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]](
      (outputs, targets) => {

        val (preds, probs) = outputs

        val sq_errors = preds.subtract(targets).square

        sq_errors.mean(axes = 1).castTo[Float]
      }
    )

  val c1 =
    DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]](
      (outputs, targets) => {

        val (preds, probs) = outputs

        val sq_errors = preds.subtract(targets).square

        probs.multiply(sq_errors).sum(axes = 1).castTo[Float]
      }
    )

  val c2 =
    DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]](
      (outputs, targets) => {

        val (preds, probs) = outputs

        val sq_errors = preds.subtract(targets).square
        val c1        = probs.multiply(sq_errors).sum(axes = 1, keepDims = true)
        probs
          .multiply(sq_errors.subtract(c1).square)
          .sum(axes = 1)
          .castTo[Float]
      }
    )

  def c1(alpha: Double, sigma_sq: Double, n: Int) =
    DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]](
      (outputs, targets) => {

        val (preds, probs) = outputs

        val sq_errors = preds.subtract(targets).square

        val one = Tensor(1d).toOutput

        val two = Tensor(2d).toOutput

        val un_p = probs * (
          tf.exp(
            tf.log(one + alpha) / two - (sq_errors * alpha) / (two * sigma_sq)
          )
        )

        //Calculate the saddle point probability
        val p = un_p / un_p.sum(axes = 1, keepDims = true)

        val c1 = p.multiply(sq_errors).sum(axes = 1).castTo[Float]

        c1
      }
    )

  def c2(alpha: Double, sigma_sq: Double, n: Int) =
    DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]](
      (outputs, targets) => {

        val (preds, probs) = outputs

        val sq_errors = preds.subtract(targets).square
        val one       = Tensor(1d).toOutput
        val two       = Tensor(2d).toOutput

        val un_p      = probs * (
          tf.exp(
            tf.log(one + alpha) / two - (sq_errors * alpha) / (two * sigma_sq)
          )
        )

        //Calculate the saddle point probability
        val p = un_p / un_p.sum(axes = 1, keepDims = true)

        val c1 = p.multiply(sq_errors).sum(axes = 1)

        p.multiply(sq_errors.subtract(c1).square)
          .sum(axes = 1)
          .castTo[Float]
      }
    )

  def apply[Pattern, In, IT, ID, IS, Loss: TF: IsFloatOrDouble](
    time_window: Int,
    modelFunction: TunableTFModel.ModelFunc[
      In,
      Output[Double],
      (Output[Double], Output[Double]),
      Loss,
      IT,
      ID,
      IS,
      Tensor[Double],
      FLOAT64,
      Shape,
      (Tensor[Double], Tensor[Double]),
      (FLOAT64, FLOAT64),
      (Shape, Shape)
    ],
    model_config_func: dtflearn.tunable_tf_model.ModelConfigFunction[
      In,
      Output[Double]
    ],
    hyp_params: Seq[String],
    persistent_hyp_params: Seq[String],
    params_to_mutable_params: Encoder[
      dtflearn.tunable_tf_model.HyperParams,
      dtflearn.tunable_tf_model.HyperParams
    ],
    training_data: DataSet[Pattern],
    tf_data_handle_ops: dtflearn.model.TFDataHandleOps[
      Pattern,
      IT,
      Tensor[Double],
      (Tensor[Double], Tensor[Double]),
      In,
      Output[Double]
    ],
    fitness_to_scalar: DataPipe[Seq[Tensor[Float]], Double] =
      DataPipe[Seq[Tensor[Float]], Double](m =>
          m.map(_.scalar.toDouble).sum / m.length),
    validation_data: Option[DataSet[Pattern]] = None,
    data_split_func: Option[DataPipe[Pattern, Boolean]] = None
  ) = new PDTModel[Pattern, In, IT, ID, IS, Loss](
    time_window,
    modelFunction,
    model_config_func,
    hyp_params,
    persistent_hyp_params,
    params_to_mutable_params,
    training_data,
    tf_data_handle_ops,
    fitness_to_scalar,
    validation_data,
    data_split_func
  )

}
