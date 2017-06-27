package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.pipes.Encoder

/**
  * Defines implicit transformations commonly used in
  * radial diffusion analysis.
  *
  * @author mandar2812 date 27/06/2017.
  * */
object implicits {

  implicit val praramsEncTuple4: Encoder[(Double, Double, Double, Double), (Double, Double, Double, Double)] =
    Encoder(
      DynaMLPipe.identityPipe[(Double, Double, Double, Double)],
      DynaMLPipe.identityPipe[(Double, Double, Double, Double)])

  implicit val paramsEncBDV: Encoder[DenseVector[Double], (Double, Double, Double, Double)] = Encoder(
    (x: DenseVector[Double]) => (x(0), x(1), x(2), x(3)),
    (x: (Double, Double, Double, Double)) => DenseVector(x._1, x._2, x._3, x._4)
  )

  implicit val paramsEncSeq: Encoder[Seq[Double], (Double, Double, Double, Double)] = Encoder(
    (x: Seq[Double]) => (x.head, x(1), x(2), x(3)),
    (x: (Double, Double, Double, Double)) => Seq(x._1, x._2, x._3, x._4)
  )

  implicit val paramsEncList: Encoder[List[Double], (Double, Double, Double, Double)] = Encoder(
    (x: List[Double]) => (x.head, x(1), x(2), x(3)),
    (x: (Double, Double, Double, Double)) => List(x._1, x._2, x._3, x._4)
  )

  implicit val paramsEncArr: Encoder[Array[Double], (Double, Double, Double, Double)] = Encoder(
    (x: Array[Double]) => (x.head, x(1), x(2), x(3)),
    (x: (Double, Double, Double, Double)) => Array(x._1, x._2, x._3, x._4)
  )

  implicit val paramsEncMap: Encoder[Map[String, Double], (Double, Double, Double, Double)] = Encoder(
    (x: Map[String, Double]) => (x("alpha"), x("beta"), x("a"), x("b")),
    (x: (Double, Double, Double, Double)) => Map(
      "alpha" -> x._1, "beta" -> x._2,
      "a" -> x._3, "b" -> x._4)
  )

  def paramsEncMap(prefix: String): Encoder[Map[String, Double], (Double, Double, Double, Double)] = Encoder(
    (x: Map[String, Double]) => (x(prefix+"/alpha"), x(prefix+"/beta"), x(prefix+"/a"), x(prefix+"/b")),
    (x: (Double, Double, Double, Double)) => Map(
      prefix+"/alpha" -> x._1, prefix+"/beta" -> x._2,
      prefix+"/a" -> x._3, prefix+"/b" -> x._4)
  )

}
