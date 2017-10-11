package io.github.mandar2812.PlasmaML.utils

import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder}

/**
  * @author mandar date 10/07/2017.
  * */
case class MagConfigEncoding(keys: (String, String, String, String)) extends
  Encoder[Map[String, Double], (Double, Double, Double, Double)] {

  override val i = DataPipe(
    (c: (Double, Double, Double, Double)) => Map(
      keys._1 -> math.log(c._1),
      keys._2 -> c._2,
      keys._3 -> c._3,
      keys._4 -> c._4
    ))

  override def run(data: Map[String, Double]) = (
    math.exp(data(keys._1)), data(keys._2),
    data(keys._3), data(keys._4))
}
