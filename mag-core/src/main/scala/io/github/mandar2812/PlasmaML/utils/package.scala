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


  /**
    * Calculate the generalised Laguerre polynomial
    * */
  def laguerre(n: Int, alpha: Double, x: Double): Double = {

    def laguerreRec(k: Int, alphav: Double, xv: Double, a: Double, b: Double): Double = k match {
      case 0 => a
      case 1 => b
      case _ => laguerreRec(k - 1, xv, alphav, ((2*k + 1 + alphav - xv)*a - (k + alphav)*b)/(k + 1), a)
    }

    laguerreRec(n, alpha, x, 1.0, 1.0 + alpha - x)
  }

  def grad_laguerre(k: Int)(n: Int, alpha: Double, x: Double): Double =
    if(k <= n) math.pow(-1.0, k)*laguerre(n - k, alpha + k, x)
    else 0d
}
