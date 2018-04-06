import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

import $file.timelag_inference_fixed

import $file.timelag_inference_const_v

import $file.timelag_inference_const_a

def main(
  fixed_lag: Float       = 3f,
  d: Int                 = 3,
  n: Int                 = 100,
  sliding_window: Int    = 15,
  noise: Double          = 0.5,
  noiserot: Double       = 0.1,
  iterations: Int        = 150000,
  optimizer: Optimizer   = tf.train.AdaDelta(0.01),
  reg: Double            = 0.01,
  p: Double              = 1.0,
  time_scale: Double     = 1.0,
  corr_sc: Double        = 2.5,
  c_cutoff: Double       = 0.0,
  prior_wt: Double       = 1d,
  mo_flag: Boolean       = false,
  prob_timelags: Boolean = false) = {

  val (res_1, res_2, res_3) = (

    timelag_inference_fixed.main(
      fixed_lag, d, n, sliding_window, noise, noiserot,
      iterations, optimizer, "const_lag", reg, p,
      time_scale, corr_sc, c_cutoff, prior_wt,
      mo_flag, prob_timelags),

    timelag_inference_const_v.main(
      d, n, sliding_window, noise, noiserot,
      iterations, optimizer, "const_v", reg, p,
      time_scale, corr_sc, c_cutoff, prior_wt,
      mo_flag, prob_timelags),

    timelag_inference_const_a.main(
      d, n, sliding_window, noise, noiserot,
      iterations, optimizer, "const_a", reg, p,
      time_scale, corr_sc, c_cutoff, prior_wt,
      mo_flag, prob_timelags)
  )

  Seq(res_1, res_2, res_3)

}