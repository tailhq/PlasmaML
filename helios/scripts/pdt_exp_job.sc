import $file.{
  pdt_const_lag_tuning => tuning_exp1,
  pdt_const_v_tuning => tuning_exp2,
  pdt_const_a_tuning => tuning_exp3,
  pdt_softplus_tuning => tuning_exp4,
  run_model_tuning_baseline => baseline_exp
}
import ammonite.ops._
import ammonite.ops.ImplicitWd._
import _root_io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import $file.env

import _root_.io.github.mandar2812.dynaml.repl.Router.main

@main
def main() = {
    
  val num_neurons_exp2 = Seq(40, 40)
  val num_neurons_exp3 = Seq(60, 40)
  val num_neurons_exp4 = Seq(60, 40)
  val act_exp2         = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f)
  val act_exp3         = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f)
  val act_exp4         = (i: Int) => timelag.utils.getReLUAct3[Double](1, 1, i, 0f)

  val exp_set2 = tuning_exp2.main(
    d = 10,
    size_training = 10000,
    size_test = 2000,
    sliding_window = 10,
    noise = 0.75,
    noiserot = 0.001,
    alpha = 0.02,
    train_test_separate = true,
    num_neurons = num_neurons_exp2,
    activation_func = act_exp2,
    iterations = 200000,
    iterations_tuning = 20000,
    pdt_iterations = 3,
    pdt_iterations_tuning = 5,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.organize_results(exp_set2, home / 'tmp / 'results_exp2, "exp2_")
  %%(
    'tar,
    "-C",
    home / 'tmp,
    "-zcvf",
    home / 'tmp / "exp2.tar.gz",
    "results_exp2"
  )

  val exp_set2_bs = baseline_exp(
    exp_set2.head.results.summary_dir,
    d = 10,
    num_neurons = num_neurons_exp2,
    activation_func = act_exp2,
    iterations = 200000,
    iterations_tuning = 20000,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.utils.write_performance_baseline(
    exp_set2_bs.head.results.metrics_train.get,
    exp_set2_bs.head.results.metrics_test.get,
    exp_set2_bs.head.results.summary_dir
  )

  val exp_set3 = tuning_exp3.main(
    d = 10,
    size_training = 10000,
    size_test = 2000,
    sliding_window = 10,
    noise = 0.75,
    noiserot = 0.001,
    alpha = 0.02,
    train_test_separate = true,
    num_neurons = num_neurons_exp3,
    activation_func = act_exp3,
    iterations = 200000,
    iterations_tuning = 20000,
    pdt_iterations = 3,
    pdt_iterations_tuning = 5,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.organize_results(exp_set3, home / 'tmp / 'results_exp3, "exp3_")
  %%(
    'tar,
    "-C",
    home / 'tmp,
    "-zcvf",
    home / 'tmp / "exp3.tar.gz",
    "results_exp3"
  )

  val exp_set3_bs = baseline_exp(
    exp_set3.head.results.summary_dir,
    num_neurons = num_neurons_exp3,
    activation_func = act_exp3,
    iterations = 200000,
    iterations_tuning = 20000,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.utils.write_performance_baseline(
    exp_set3_bs.head.results.metrics_train.get,
    exp_set3_bs.head.results.metrics_test.get,
    exp_set3_bs.head.results.summary_dir
  )

  val exp_set4 = tuning_exp4.main(
    d = 10,
    size_training = 10000,
    size_test = 2000,
    sliding_window = 10,
    noise = 0.75,
    noiserot = 0.001,
    alpha = 0.02,
    train_test_separate = true,
    num_neurons = num_neurons_exp4,
    activation_func = act_exp4,
    iterations = 200000,
    iterations_tuning = 20000,
    pdt_iterations = 3,
    pdt_iterations_tuning = 5,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.organize_results(exp_set4, home / 'tmp / 'results_exp4, "exp4_")
  %%(
    'tar,
    "-C",
    home / 'tmp,
    "-zcvf",
    home / 'tmp / "exp4.tar.gz",
    "results_exp4"
  )

  val exp_set4_bs = baseline_exp(
    exp_set4.head.results.summary_dir,
    num_neurons = num_neurons_exp4,
    activation_func = act_exp4,
    iterations = 100000,
    iterations_tuning = 20000,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.utils.write_performance_baseline(
    exp_set4_bs.head.results.metrics_train.get,
    exp_set4_bs.head.results.metrics_test.get,
    exp_set4_bs.head.results.summary_dir
  )

  val exp_set1 = tuning_exp1.main(
    d = 10,
    size_training = 8000,
    size_test = 2000,
    sliding_window = 20,
    noise = 0.7,
    noiserot = 0.001,
    alpha = 0.02,
    train_test_separate = true,
    num_neurons = num_neurons_exp2,
    activation_func = act_exp2,
    iterations = 60000,
    iterations_tuning = 10000,
    pdt_iterations = 3,
    pdt_iterations_tuning = 5,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.AdaDelta(0.01f),
    confounding = Seq(0d),
    num_samples = 20,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2")
  )

  timelag.organize_results(exp_set1, home / 'tmp / 'results_exp1, "exp1_")
  %%(
    'tar,
    "-C",
    home / 'tmp,
    "-zcvf",
    home / 'tmp / "exp1.tar.gz",
    "results_exp1"
  )

  val exp_set1_bs = baseline_exp(
    exp_set1.head.results.summary_dir,
    num_neurons = num_neurons_exp2,
    activation_func = act_exp2,
    iterations = 100000,
    iterations_tuning = 20000,
    miniBatch = 128,
    optimizer = org.platanios.tensorflow.api.tf.train.Adam(0.01f),
    confounding = Seq(0d),
    num_samples = 4,
    hyper_optimizer = "gs",
    hyp_opt_iterations = Some(8),
    regularization_types = Seq("L2"),
    checkpointing_freq = 1
  )

  timelag.utils.write_performance_baseline(
    exp_set1_bs.head.results.metrics_train.get,
    exp_set1_bs.head.results.metrics_test.get,
    exp_set1_bs.head.results.summary_dir
  )
}
