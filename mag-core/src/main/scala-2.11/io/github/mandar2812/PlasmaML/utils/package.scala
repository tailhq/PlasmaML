package io.github.mandar2812.PlasmaML

import breeze.linalg._
import io.github.mandar2812.dynaml.pipes._

/**
  * Miscellaneous utilities for the PlasmaML software distribution.
  *
  * @author mandar2812 date 16/05/2017.
  * */
package object utils {


  /**
    * Generates a fourier series feature mapping.
    * */
  object FourierSeriesGenerator extends MetaPipe21[Double, Int, Double, DenseVector[Double]] {

    override def run(omega: Double, components: Int): DataPipe[Double, DenseVector[Double]] = {

      DataPipe((x: Double) => DenseVector((0 to components).map(i => if(i == 0) 1d else math.sin(i*omega*x)).toArray))
    }

  }

  /**
    * Generates a polynomial feature mapping upto a specified degree.
    * */
  object PolynomialSeriesGenerator extends MetaPipe[Int, Double, DenseVector[Double]] {

    override def run(degree: Int): DataPipe[Double, DenseVector[Double]] = {

      DataPipe((x: Double) => DenseVector((0 to degree).map(d => math.pow(x, d)).toArray))
    }
  }

}
