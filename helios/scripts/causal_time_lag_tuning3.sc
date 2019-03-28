import $exec.helios.scripts.{
    tl_const_v_tuning => tuning_exp2, 
    tl_const_a_tuning => tuning_exp3, 
    tl_softplus_tuning => tuning_exp4}
  
import ammonite.ops.ImplicitWd._
    
import org.platanios.tensorflow.api.learn.layers.Activation

val getReLUAct2: Int => Int => Activation = (s: Int) => (i: Int) => {
  if((i - s) == 0) tf.learn.ReLU(s"Act_$i", 0.01f)
  else tf.learn.Sigmoid(s"Act_$i")
}

val exp_set2 = tuning_exp2.main(
  d = 10, size_training = 8000, size_test = 2000, 
  sliding_window = 20, 
  noise = 0.7, noiserot = 0.001,
  alpha = 0.02, train_test_separate = true,
  num_neurons = Seq(30, 25, 20), activation_func = getReLUAct2(1),
  iterations = 150000, iterations_tuning = 10000, 
  miniBatch = 1024, optimizer = tf.train.AdaDelta(0.01), 
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler, helios.learn.cdt_loss.Hellinger), 
  target_prob = Seq(helios.learn.cdt_loss.Boltzmann, helios.learn.cdt_loss.Uniform),
  dist_type = "default", num_samples = 20, 
  hyper_optimizer = "gs", 
  hyp_opt_iterations = Some(8), 
  regularization_types = Seq("L2"))
  
  
timelag.organize_results(exp_set2, home/'tmp/'results_3l_exp2)
%%('tar, "-C", home/'tmp, "-zcvf", home/'tmp/"exp2_3l.tar.gz", "results_3l_exp2") 
  
val exp_set3 = tuning_exp3.main(
  d = 10, size_training = 8000, size_test = 2000, 
  sliding_window = 20, 
  noise = 0.7, noiserot = 0.001,
  alpha = 0.02, train_test_separate = true,
  num_neurons = Seq(30, 25, 20), activation_func = getReLUAct2(1),
  iterations = 150000, iterations_tuning = 10000, 
  miniBatch = 1024, optimizer = tf.train.AdaDelta(0.01), 
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler, helios.learn.cdt_loss.Hellinger), 
  target_prob = Seq(helios.learn.cdt_loss.Boltzmann, helios.learn.cdt_loss.Uniform),
  dist_type = "default", num_samples = 20, 
  hyper_optimizer = "gs", 
  hyp_opt_iterations = Some(8), 
  regularization_types = Seq("L2"))
  
  
timelag.organize_results(exp_set3, home/'tmp/'results_3l_exp3)  
%%('tar, "-C", home/'tmp, "-zcvf", home/'tmp/"exp3_3l.tar.gz", "results_3l_exp3") 

  
val exp_set4 = tuning_exp4.main(
  d = 10, 
  size_training = 8000, size_test = 2000, 
  sliding_window = 20,
  noise = 0.7, noiserot = 0.001,
  alpha = 0.02, train_test_separate = true,
  num_neurons = Seq(30, 25, 20), activation_func = getReLUAct2(1),
  iterations = 150000, iterations_tuning = 10000, 
  miniBatch = 1024, optimizer = tf.train.AdaDelta(0.01), 
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler, helios.learn.cdt_loss.Hellinger), 
  target_prob = Seq(helios.learn.cdt_loss.Boltzmann, helios.learn.cdt_loss.Uniform),
  dist_type = "default", num_samples = 20, 
  hyper_optimizer = "gs", 
  hyp_opt_iterations = Some(8), 
  regularization_types = Seq("L2"))
  
  
timelag.organize_results(exp_set4, home/'tmp/'results_3l_exp4)  
%%('tar, "-C", home/'tmp, "-zcvf", home/'tmp/"exp4_3l.tar.gz", "results_3l_exp4") 
