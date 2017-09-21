package io.github.mandar2812.PlasmaML.utils

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.kernels.{KernelMatrix, LocalSVMKernel, SVMKernelMatrix}

class DiracTuple2Kernel(private var noiseLevel: Double = 1.0)
  extends LocalSVMKernel[(Double, Double)]
    with Serializable {

  override val hyper_parameters = List("noiseLevel")

  state = Map("noiseLevel" -> noiseLevel)

  def setNoiseLevel(d: Double): Unit = {
    this.state += ("noiseLevel" -> d)
    this.noiseLevel = d
  }

  override def evaluateAt(
    config: Map[String, Double])(
    x: (Double, Double), y: (Double, Double)): Double =
    if (x._1 == y._1 && x._2 == y._2) math.abs(config("noiseLevel"))*1.0 else 0.0

  override def gradientAt(
    config: Map[String, Double])(
    x: (Double, Double), y: (Double, Double)): Map[String, Double] =
    Map("noiseLevel" -> 1.0*evaluateAt(config)(x,y)/math.abs(config("noiseLevel")))

  /*override def buildKernelMatrix[S <: Seq[(Double, Double)]](
    mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
    new SVMKernelMatrix(DenseMatrix.eye[Double](length)*state("noiseLevel"), length)*/

}
