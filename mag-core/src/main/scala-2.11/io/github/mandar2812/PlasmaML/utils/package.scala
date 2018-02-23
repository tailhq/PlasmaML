package io.github.mandar2812.PlasmaML

/**
  * Miscellaneous utilities for the PlasmaML software distribution.
  *
  * @author mandar2812 date 16/05/2017.
  * */
package object utils {

  def mean(seq: Seq[Double]): Double = seq.sum/seq.length

  def generalized_logistic(
    lower: Double, upper: Double, rate: Double,
    start_time: Double, µ: Double, q: Double,
    c: Double)(t: Double) = {
    require(µ > 0d, "In a generalized logistic curve, growth order must be positive")
    lower + (upper - lower)/math.pow(c + q*math.exp(-rate*(t - start_time)), 1d/µ)
  }

}
