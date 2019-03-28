import $exec.helios.scripts.{
  tl_const_v_tuning => tuning_exp2, 
  tl_const_a_tuning => tuning_exp3, 
  tl_softplus_tuning => tuning_exp4}


val res_tune_exp2 = tuning_exp2.main(
  d = 10, size_training = 8000, size_test = 2000, 
  sliding_window = 20, 
  noise = 0.7, noiserot = 0.001,
  alpha = 0.02, train_test_separate = true,
  num_neurons = Seq(30, 25), 
  iterations = 150000, iterations_tuning = 10000, 
  miniBatch = 1024, optimizer = tf.train.AdaDelta(0.01), 
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler), 
  target_prob = Seq(helios.learn.cdt_loss.Uniform, helios.learn.cdt_loss.Boltzmann),
  dist_type = "default", num_samples = 20, 
  hyper_optimizer = "gs", 
  hyp_opt_iterations = Some(8), 
  regularization_type = "L2")


timelag.organize_results(res_tune_exp2, home/"tmp"/"results_exp2")  

val res_tune_exp3 = tuning_exp3.main(
  d = 10, size_training = 8000, size_test = 2000, 
  sliding_window = 20, 
  noise = 0.7, noiserot = 0.001,
  alpha = 0.02, train_test_separate = true,
  num_neurons = Seq(30, 25), 
  iterations = 150000, iterations_tuning = 10000, 
  miniBatch = 1024, optimizer = tf.train.AdaDelta(0.01), 
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler), 
  target_prob = Seq(helios.learn.cdt_loss.Uniform, helios.learn.cdt_loss.Boltzmann),
  dist_type = "default", num_samples = 20, 
  hyper_optimizer = "gs", 
  hyp_opt_iterations = Some(8), 
  regularization_type = "L2")


timelag.organize_results(res_tune_exp3, home/"tmp"/"results_exp3")  


val res_tune_exp4 = tuning_exp4.main(
  d = 10, 
  size_training = 8000, size_test = 2000, 
  sliding_window = 20,
  noise = 0.7, noiserot = 0.001,
  alpha = 0.02, train_test_separate = true,
  num_neurons = Seq(30, 25), 
  iterations = 150000, iterations_tuning = 10000, 
  miniBatch = 1024, optimizer = tf.train.AdaDelta(0.01), 
  prior_type = Seq(helios.learn.cdt_loss.KullbackLeibler), 
  target_prob = Seq(helios.learn.cdt_loss.Uniform, helios.learn.cdt_loss.Boltzmann),
  dist_type = "default", num_samples = 20, 
  hyper_optimizer = "gs", 
  hyp_opt_iterations = Some(8), 
  regularization_type = "L2")


timelag.organize_results(res_tune_exp4, home/"tmp"/"results_exp4")  
