import $exec.helios.scripts.ctl_tuning_solar_images
import $exec.helios.scripts.env

val exp_2015 = ctl_tuning_solar_images(
  year_range = 2011 to 2017,
  test_year = 2015,
  ff_stack_sizes = Seq(256, 128, 64),
  image_source = SDO(HMIB, 512),
  re = false,
  scaleDown = 4,
  image_hist = 96,
  image_hist_downsamp = 12,
  opt = tf.train.AdaDelta(0.001f),
  iterations = 500000,
  iterations_tuning = 50000,
  miniBatch = 16,
  num_hyp_samples = 10,
  hyper_optimizer = "gs",
  path_to_images = Some(env.data_dir),
  tmpdir = env.summary_dir
)
