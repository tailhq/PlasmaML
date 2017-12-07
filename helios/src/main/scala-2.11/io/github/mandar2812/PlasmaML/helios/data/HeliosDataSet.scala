package io.github.mandar2812.PlasmaML.helios.data

import org.platanios.tensorflow.api._

case class HeliosDataSet(
  trainData: Tensor, trainLabels: Tensor,
  testData: Tensor, testLabels: Tensor)
