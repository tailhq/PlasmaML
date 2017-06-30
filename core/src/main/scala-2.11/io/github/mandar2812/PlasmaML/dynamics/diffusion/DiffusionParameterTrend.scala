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
  val transform: Encoder[T, (Double, Double, Double, Double)])
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

object DiffusionParameterTrend {

  def getEncoder(prefix: String) = Encoder(
    (c: Map[String, Double]) => (
      c(prefix+"_alpha"),
      c(prefix+"_beta"),
      c(prefix+"_a"),
      c(prefix+"_b")),
    (p: (Double, Double, Double, Double)) => Map(
      prefix+"_alpha" -> p._1,
      prefix+"_beta" -> p._2,
      prefix+"_a" -> p._3,
      prefix+"_b" -> p._4)
  )

}