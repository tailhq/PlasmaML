import $exec.helios.scripts.{
  pdt_const_lag_tuning => tuning_exp1,
  pdt_const_v_tuning => tuning_exp2,
  pdt_const_a_tuning => tuning_exp3,
  pdt_softplus_tuning => tuning_exp4
}

import ammonite.ops.ImplicitWd._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import $exec.helios.scripts.env

val exp_set1 = tuning_exp1.main(
  d = 10,
  size_training = 8000,
  size_test = 2000,
  sliding_window = 20,
  noise = 0.7,
  noiserot = 0.001,
  alpha = 0.02,
  train_test_separate = true,
  num_neurons = Seq(30, 25),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  iterations = 150000,
  iterations_tuning = 10000,
  miniBatch = 1024,
  optimizer = tf.train.AdaDelta(0.01f),
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_prob =
    Seq(helios.learn.cdt_loss.Boltzmann, helios.learn.cdt_loss.Uniform),
  confounding = Seq(0d),
  dist_type = "default",
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

val exp_set2 = tuning_exp2.main(
  d = 10,
  size_training = 10000,
  size_test = 2000,
  sliding_window = 20,
  noise = 0.7,
  noiserot = 0.001,
  alpha = 0.02,
  train_test_separate = true,
  num_neurons = Seq(20, 20, 20),
  activation_func = (i: Int) => timelag.utils.getReLUAct3[Double](3, i),
  iterations = 10000,
  iterations_tuning = 5000,
  miniBatch = 1024,
  optimizer = tf.train.Adam(0.001f),
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_prob = Seq(helios.learn.cdt_loss.Boltzmann),
  confounding = Seq(0d),
  dist_type = "default",
  num_samples = 20,
  hyper_optimizer = "gs",
  hyp_opt_iterations = Some(8),
  regularization_types = Seq("L2")
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

val exp_set3 = tuning_exp3.main(
  d = 10,
  size_training = 10000,
  size_test = 2000,
  sliding_window = 20,
  noise = 0.7,
  noiserot = 0.001,
  alpha = 0.02,
  train_test_separate = true,
  num_neurons = Seq(40, 40),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  iterations = 20000,
  iterations_tuning = 5000,
  miniBatch = 1024,
  optimizer = tf.train.Adam(0.01f),
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_prob = Seq(helios.learn.cdt_loss.Boltzmann),
  confounding = Seq(0d),
  dist_type = "default",
  num_samples = 20,
  hyper_optimizer = "gs",
  hyp_opt_iterations = Some(8),
  regularization_types = Seq("L2")
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

val exp_set4 = tuning_exp4.main(
  d = 10,
  size_training = 10000,
  size_test = 2000,
  sliding_window = 20,
  noise = 0.7,
  noiserot = 0.001,
  alpha = 0.02,
  train_test_separate = true,
  num_neurons = Seq(60, 40),
  activation_func = (i: Int) => timelag.utils.getReLUAct2[Double](1, i),
  iterations = 50000,
  iterations_tuning = 5000,
  miniBatch = 1024,
  optimizer = tf.train.Adam(0.01f),
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler),
  target_prob = Seq(helios.learn.cdt_loss.Boltzmann),
  confounding = Seq(0d),
  dist_type = "default",
  num_samples = 20,
  hyper_optimizer = "gs",
  hyp_opt_iterations = Some(8),
  regularization_types = Seq("L2")
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
