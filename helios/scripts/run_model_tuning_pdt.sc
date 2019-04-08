import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.pipes._
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

@main
def apply(
  compute_output: DataPipe[Tensor[Double], (Float, Float)],
  d: Int = 10,
  confounding: Seq[Double] = Seq(0d, 0.25, 0.5, 0.75),
  size_training: Int = 1000,
  size_test: Int = 500,
  sliding_window: Int = 15,
  noise: Double = 0.5,
  noiserot: Double = 0.1,
  alpha: Double = 0.0,
  train_test_separate: Boolean = false,
  num_neurons: Seq[Int] = Seq(40),
  activation_func: Int => Activation[Double] = (i: Int) =>
    timelag.utils.getReLUAct2[Double](1, i),
  iterations: Int = 150000,
  iterations_tuning: Int = 20000,
  miniBatch: Int = 32,
  optimizer: Optimizer = tf.train.AdaDelta(0.01f),
  sum_dir_prefix: String = "cdt",
  prior_types: Seq[helios.learn.cdt_loss.Divergence] =
    Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_probs: Seq[helios.learn.cdt_loss.TargetDistribution] =
    Seq(helios.learn.cdt_loss.Boltzmann),
  dist_type: String = "default",
  timelag_pred_strategy: String = "mode",
  summaries_top_dir: Path = home / 'tmp,
  num_samples: Int = 20,
  hyper_optimizer: String = "gs",
  hyp_opt_iterations: Option[Int] = Some(5),
  epochFlag: Boolean = false,
  regularization_types: Seq[String] = Seq("L2")
): Seq[timelag.ExperimentResult[Double, Double, timelag.TunedModelRun[
  Double,
  Double
]]] = {

  val mo_flag       = true
  val prob_timelags = true

  val num_pred_dims = timelag.utils.get_num_output_dims(
    sliding_window,
    mo_flag,
    prob_timelags,
    dist_type
  )

  val (net_layer_sizes, layer_shapes, layer_parameter_names, layer_datatypes) =
    timelag.utils.get_ffnet_properties(
      -1,
      num_pred_dims,
      num_neurons,
      "FLOAT64"
    )

  val output_mapping = timelag.utils.get_output_mapping[Double](
    sliding_window,
    mo_flag,
    prob_timelags,
    dist_type
  )

  //Prediction architecture
  val architecture =
    dtflearn.feedforward_stack[Double](activation_func)(net_layer_sizes.tail) >>
      output_mapping

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

  val hyper_parameters = List(
    "temperature",
    "reg"
  )

  val hyper_prior = Map(
    "temperature" -> UniformRV(1d, 2.0),
    "reg"         -> UniformRV(-5d, -3d)
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

  val fitness_function =
    DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]](
      (preds, targets) => {

        val weighted_error = preds._1
          .subtract(targets)
          .square
          .multiply(preds._2)
          .sum(axes = 1)

        val entropy = preds._2
          .multiply(Tensor(-1d).castTo[Double])
          .multiply(tf.log(preds._2))
          .sum(axes = 1)

        (weighted_error + entropy).castTo[Float]
      }
    )

  val dataset: timelag.utils.TLDATA[Double] =
    timelag.utils.generate_data[Double](
      compute_output,
      sliding_window,
      d,
      size_training,
      noiserot,
      alpha,
      noise
    )

  val dataset_test: timelag.utils.TLDATA[Double] =
    timelag.utils.generate_data[Double](
      compute_output,
      sliding_window,
      d,
      size_test,
      noiserot,
      alpha,
      noise
    )

  for (c                   <- confounding;
       prior_type          <- prior_types;
       target_prob         <- target_probs;
       regularization_type <- regularization_types) yield {

    val loss_func_generator = (h: Map[String, Double]) => {

      val lossFunc = timelag.utils.get_pdt_loss[Double, Double, Double](
        sliding_window,
        temp = h("temperature"), 
        target_prob
      )

      val reg_layer =
        if (regularization_type == "L1")
          L1Regularization[Double](
            layer_scopes,
            layer_parameter_names,
            layer_datatypes,
            layer_shapes,
            math.exp(h("reg")),
            "L1Reg"
          )
        else
          L2Regularization[Double](
            layer_scopes,
            layer_parameter_names,
            layer_datatypes,
            layer_shapes,
            math.exp(h("reg")),
            "L2Reg"
          )

      lossFunc >>
        reg_layer >>
        tf.learn.ScalarSummary("Loss", "ModelLoss")
    }

    val result = timelag.run_exp_hyp(
      (dataset, dataset_test),
      architecture,
      hyper_parameters,
      loss_func_generator,
      fitness_function,
      hyper_prior,
      iterations,
      iterations_tuning,
      optimizer,
      miniBatch,
      sum_dir_prefix,
      mo_flag,
      prob_timelags,
      timelag_pred_strategy,
      summaries_top_dir,
      num_samples,
      hyper_optimizer,
      hyp_opt_iterations = hyp_opt_iterations,
      epochFlag = epochFlag,
      hyp_mapping = hyp_mapping,
      confounding_factor = c
    )

    result.copy[Double, Double, timelag.TunedModelRun[Double, Double]](
      config = result.config.copy[Double](
        divergence = Some(prior_type),
        target_prob = Some(target_prob),
        reg_type = Some(regularization_type)
      )
    )
  }

}
