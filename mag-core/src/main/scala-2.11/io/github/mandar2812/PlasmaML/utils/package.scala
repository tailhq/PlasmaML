package io.github.mandar2812.PlasmaML

/**
  * Miscellaneous utilities for the PlasmaML software distribution.
  *
  * @author mandar2812 date 16/05/2017.
  * */
package object utils {

  def mean(seq: Seq[Double]): Double = seq.sum/seq.length

}
