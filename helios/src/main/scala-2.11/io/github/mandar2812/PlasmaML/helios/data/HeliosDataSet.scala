package io.github.mandar2812.PlasmaML.helios.data

import org.platanios.tensorflow.api._

case class HeliosDataSet(
  trainData: Tensor, trainLabels: Tensor, nTrain: Int,
  testData: Tensor, testLabels: Tensor, nTest: Int) {


  def close(): Unit = {
    trainData.close()
    trainLabels.close()
    testData.close()
    testLabels.close()
  }
}


case class AbstractDataSet[TI, TT](
  trainData: TI, trainLabels: TT, nTrain: Int,
  testData: TI, testLabels: TT, nTest: Int)