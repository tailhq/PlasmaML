package io.github.mandar2812.PlasmaML.utils

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder}

/**
  * Created by mandar on 16/05/2017.
  */
case class ConfigEncoding(keys: List[String]) extends Encoder[Map[String, Double], DenseVector[Double]] {
  self =>

  override val i = DataPipe((x: DenseVector[Double]) => keys.zip(x.toArray).toMap)

  override def run(data: Map[String, Double]) = DenseVector(keys.map(data(_)).toArray)

  def reverseEncoder = Encoder(i, DataPipe(self.run _))
}
