package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SamePadding

/**
  * A collection of architectures building blocks
  * for learning and predicting from solar images.
  *
  * */
object Arch {

  /**
    * CNN architecture for GOES XRay flux (short wavelength)
    * */
  val cnn_goes_v1 = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      tf.learn.Conv2D("Conv2D_0", Shape(2, 2, 4, 64), 1, 1, SamePadding) >>
      tf.learn.AddBias(name = "Bias_0") >>
      tf.learn.ReLU("ReLU_0", 0.1f) >>
      tf.learn.Dropout("Dropout_0", 0.6f) >>
      tf.learn.Conv2D("Conv2D_1", Shape(2, 2, 64, 32), 2, 2, SamePadding) >>
      tf.learn.AddBias(name = "Bias_1") >>
      tf.learn.ReLU("ReLU_1", 0.1f) >>
      tf.learn.Dropout("Dropout_1", 0.6f) >>
      tf.learn.Conv2D("Conv2D_2", Shape(2, 2, 32, 16), 4, 4, SamePadding) >>
      tf.learn.AddBias(name = "Bias_2") >>
      tf.learn.ReLU("ReLU_2", 0.1f) >>
      tf.learn.Dropout("Dropout_2", 0.6f) >>
      tf.learn.Conv2D("Conv2D_3", Shape(2, 2, 16, 8), 8, 8, SamePadding) >>
      tf.learn.AddBias(name = "Bias_3") >>
      tf.learn.ReLU("ReLU_3", 0.1f) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
      tf.learn.Flatten("Flatten_3") >>
      tf.learn.Linear("FC_Layer_4", 128) >>
      tf.learn.ReLU("ReLU_4", 0.1f) >>
      tf.learn.Linear("FC_Layer_5", 64) >>
      tf.learn.ReLU("ReLU_5", 0.1f) >>
      tf.learn.Linear("FC_Layer_6", 8) >>
      tf.learn.Sigmoid("Sigmoid_6") >>
      tf.learn.Linear("OutputLayer", 1)
  }



}
