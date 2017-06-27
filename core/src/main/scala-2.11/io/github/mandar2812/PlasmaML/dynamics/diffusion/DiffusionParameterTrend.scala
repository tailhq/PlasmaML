package io.github.mandar2812.PlasmaML.dynamics.diffusion

import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder, MetaPipe}

/**
  * Specifies a trend function for radial diffusion priors.
  *
  * h(l, t) = &alpha; l<sup>&beta;</sup 10<sup>a + b Kp(t)</sup>
  *
  * @author mandar2812 date 27/06/2017.
  * */
class DiffusionParameterTrend[T](val Kp: DataPipe[Double, Double])(
  implicit val transform: Encoder[T, (Double, Double, Double, Double)])
  extends MetaPipe[T, (Double, Double), Double] {


  override def run(data: T) = {
    val (alpha, beta, a, b) = transform(data)

    DataPipe((lt: (Double, Double)) => {
      val (l, t) = lt

      val kp = Kp(t)
      alpha*math.pow(l, beta)*math.pow(10d, a + b*kp)
    })

  }
}
