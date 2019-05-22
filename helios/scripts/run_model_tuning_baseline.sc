import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.DynaMLPipe._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.dynaml.probability._
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
import org.platanios.tensorflow.api.learn.Mode

@main
def apply(
  experiment: Path,
  d: Int,
  confounding: Seq[Double] = Seq(0d, 0.25, 0.5, 0.75),
  num_neurons: Seq[Int] = Seq(40),
  activation_func: Int => Layer[Output[Double], Output[Double]] = (i: Int) =>
    timelag.utils.getReLUAct2[Double](1, i),
  iterations: Int = 150000,
  iterations_tuning: Int = 20000,
  miniBatch: Int = 32,
  optimizer: Optimizer = tf.train.AdaDelta(0.01f),
  num_samples: Int = 20,
  hyper_optimizer: String = "gs",
  hyp_opt_iterations: Option[Int] = Some(5),
  epochFlag: Boolean = false,
  regularization_types: Seq[String] = Seq("L2"),
  checkpointing_freq: Int = 4
): Seq[timelag.ExperimentResult2[Double, Double, timelag.RESULTBL[Double, Double]]] = {


  val num_pred_dims = 1

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      d,
      num_pred_dims,
      num_neurons,
      "FLOAT64"
    )

  //Prediction architecture
  val architecture =
    dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail)

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

  val hyper_parameters = List(
    "reg"
  )

  val hyper_prior = Map(
    "reg" -> UniformRV(-5.5d, -4d)
  )

  val logit =
    Encoder((x: Double) => math.log(x / (1d - x)), (x: Double) => sigmoid(x))

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

  val hyp_mapping = Some(
    hyper_parameters
      .map(
        h => (h, hyp_scaling(h) > logit)
      )
      .toMap
  )

  val fitness_func = Seq(
    DataPipe2[Output[Double], Output[Double], Output[Float]](
      (p, t) => p.subtract(t).square.sum(axes = -1).mean().castTo[Float]
    ),
    DataPipe2[Output[Double], Output[Double], Output[Float]](
      (p, t) => p.subtract(t).abs.sum(axes = -1).mean().castTo[Float]
    )
  )

  val fitness_to_scalar =
    DataPipe[Seq[Tensor[Float]], Double](s => {
      val metrics = s.map(_.scalar.toDouble)
      metrics.sum / metrics.length
    })

  for (c                   <- confounding;
       regularization_type <- regularization_types) yield {

    val loss_func_generator = (h: Map[String, Double]) => {

      val lossFunc = tf.learn.L2Loss[Double, Double]("Loss/L2Error")

      val reg =
        if (regularization_type == "L2")
          L2Regularization[Double](
            layer_scopes,
            layer_parameter_names,
            layer_datatypes,
            layer_shapes,
            math.exp(h("reg")),
            "L2Reg"
          )
        else
          L1Regularization[Double](
            layer_scopes,
            layer_parameter_names,
            layer_datatypes,
            layer_shapes,
            math.exp(h("reg")),
            "L1Reg"
          )

      lossFunc >>
        tf.learn.Mean[Double]("Loss/Mean") >>
        reg >>
        tf.learn.ScalarSummary("Loss", "ModelLoss")
    }

    val result = timelag.run_exp_baseline(
      experiment,
      architecture,
      hyper_parameters,
      loss_func_generator,
      fitness_func,
      hyper_prior,
      iterations,
      iterations_tuning,
      optimizer,
      miniBatch,
      num_samples,
      hyper_optimizer,
      hyp_opt_iterations = hyp_opt_iterations,
      epochFlag = epochFlag,
      hyp_mapping = hyp_mapping,
      confounding_factor = c,
      fitness_to_scalar = fitness_to_scalar,
      checkpointing_freq = checkpointing_freq
    )

    result.copy[Double, Double, timelag.RESULTBL[Double, Double]](
      config = result.config.copy[Double](
        divergence = None,
        target_prob = None,
        reg_type = Some(regularization_type)
      )
    )
  }

}
