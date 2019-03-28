package io.github.mandar2812.PlasmaML.helios.data

import org.platanios.tensorflow.api._

case class HeliosDataSet[T: TF, U: TF](
  trainData: Tensor[T], trainLabels: Tensor[U], nTrain: Int,
  testData: Tensor[T], testLabels: Tensor[U], nTest: Int) {


  def close(): Unit = {
    trainData.close()
    trainLabels.close()
    testData.close()
    testLabels.close()
  }
}
