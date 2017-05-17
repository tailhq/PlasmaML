package io.github.mandar2812.PlasmaML

import breeze.stats.distributions.ContinuousDistr

/**
  * Miscellaneous utilities for the PlasmaML software distribution.
  *
  * @author mandar2812 date 16/05/2017.
  * */
package object utils {

  def getPriorMapDistr(d: Map[String, ContinuousDistr[Double]]) = {


    new ContinuousDistr[Map[String, Double]] {

      override def unnormalizedLogPdf(x: Map[String, Double]) = {

        x.map(c => d(c._1).unnormalizedLogPdf(c._2)).sum
      }

      override def logNormalizer = d.values.map(_.logNormalizer).sum

      override def draw() = d.mapValues(_.draw())
    }

  }

}
