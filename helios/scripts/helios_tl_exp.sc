import $exec.helios.scripts.{
  timelagutils,
  timelag_inference_fixed => exp1,
  timelag_inference_const_v => exp2,
  timelag_inference_const_a => exp3,
  timelag_inference_softplus => exp4}


val res_exp2 = exp2.main(
  d = 8, n = 4000, sliding_window = 20,
  noise = 0.5, noiserot = 0.001,
  alpha = 0.005, train_test_separate = true,
  num_neurons = Seq(40, 20), 
  iterations = 150000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.01),
  reg = 0.001, c = 1.0, 
  prior_type = helios.learn.cdt_loss.JensenShannon, 
  error_wt = 1.0, prior_wt = 0.75,
  mo_flag = true, prob_timelags = true,
  dist_type = "default")

val res_exp2_sw = exp2.stage_wise(
  d = 8, n = 4000, sliding_window = 20,
  noise = 0.5, noiserot = 0.001,
  alpha = 0.005, 
  num_neurons_i = Seq(40, 20),
  num_neurons_ii = Seq(30, 20), 
  iterations = 150000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.01),
  reg_i = 0.001, reg_ii = 0.0001, c = 1.0,
  prior_type = helios.learn.cdt_loss.JensenShannon, 
  error_wt = 1.0, prior_wt = 0.75,
  mo_flag = true, prob_timelags = true,
  dist_type = "default")


val res_exp3 = exp3.main(
  d = 8, n = 4000, sliding_window = 20,
  noise = 0.65, noiserot = 0.001,
  alpha = 0.005, train_test_separate = true,
  num_neurons = Seq(40, 25),
  iterations = 150000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.01),
  reg = 0.001, c = 1.0, 
  prior_type = "Kullback-Leibler",
  error_wt = 1.0, prior_wt = 0.75,
  mo_flag = true, prob_timelags = true,
  dist_type = "default")


val res_exp1 = exp1.main(
  d = 8, n = 4000, sliding_window = 20,
  noise = 0.25, noiserot = 0.001,
  alpha = 0.0035, train_test_separate = true,
  num_neurons = Seq(15),
  iterations = 120000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.001),
  reg = 0.0001, prior_wt = 0.8,
  prior_type = "Hellinger",
  temp = 0.75, error_wt = 1.0,
  mo_flag = true, prob_timelags = true)


val res_exp4 = exp4.main(
  d = 8, n = 4000, sliding_window = 20,
  noise = 0.5, noiserot = 0.001,
  alpha = 0.0045, train_test_separate = true,
  num_neurons = Seq(20, 20),
  iterations = 25000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.001),
  reg = 0.0001, c = 4.0,
  prior_type = "Kullback-Leibler",
  error_wt = 1.0, prior_wt = 0.75,
  mo_flag = true, prob_timelags = true,
  dist_type = "default")


val res_exp2_32 = exp2.main(
  d = 32, n = 4000, sliding_window = 15,
  noise = 1.1, noiserot = 0.001,
  alpha = 0.007, train_test_separate = true,
  num_neurons = Seq(40, 30),
  iterations = 120000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.01),
  reg = 0.01, prior_wt = 1.0,
  prior_type = "Hellinger",
  temp = 0.75, error_wt = 1.0,
  mo_flag = true, prob_timelags = true,
  dist_type = "default")

val res_exp3_32 = exp3.main(
  d = 32, n = 4000, sliding_window = 15,
  noise = 1.1, noiserot = 0.001,
  alpha = 0.007, train_test_separate = true,
  num_neurons = Seq(40, 30),
  iterations = 120000, miniBatch = 1024,
  optimizer = tf.train.Adam(0.01),
  reg = 0.01, prior_wt = 1.0,
  prior_type = "Hellinger",
  temp = 0.75, error_wt = 1.0,
  mo_flag = true, dist_type = "default")
