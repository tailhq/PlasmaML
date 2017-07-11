package io.github.mandar2812.PlasmaML.dynamics.diffusion

import io.github.mandar2812.PlasmaML.utils.MagConfigEncoding
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder, MetaPipe}

/**
  * Specifies a trend function for radial diffusion priors.
  *
  * h(l, t) = &alpha; l<sup>&beta;</sup 10<sup>a + b Kp(t)</sup>
  *
  * @author mandar2812 date 27/06/2017.
  * */
class MagnetosphericProcessTrend[T](val Kp: DataPipe[Double, Double])(
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

object MagnetosphericProcessTrend {

  def getEncoder(prefix: String): MagConfigEncoding = MagConfigEncoding(
    (prefix+"_alpha", prefix+"_beta", prefix+"_a", prefix+"_b")
  )

}

class MagTrend(override val Kp: DataPipe[Double, Double], prefix: String) extends
  MagnetosphericProcessTrend[Map[String, Double]](Kp)(MagnetosphericProcessTrend.getEncoder(prefix)) {

  override val transform: MagConfigEncoding = MagnetosphericProcessTrend.getEncoder(prefix)

}


class SimpleMagTrend(prefix: String) extends MagTrend(DataPipe((_: Double) => 0d), prefix)