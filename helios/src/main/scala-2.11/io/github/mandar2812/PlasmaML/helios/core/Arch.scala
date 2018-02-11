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
    * Constructs a convolutional layer activated by a ReLU, with
    * an option of appending a dropout layer.
    *
    * */
  def conv2d_unit(
    shape: Shape, stride: (Int, Int) = (1, 1),
    relu_param: Float = 0.1f, dropout: Boolean = true,
    keep_prob: Float = 0.6f)(i: Int) =
    if(dropout) {
      tf.learn.Conv2D("Conv2D_"+i, shape, stride._1, stride._2, SamePadding) >>
        tf.learn.AddBias(name = "Bias_"+i) >>
        tf.learn.ReLU("ReLU_"+i, relu_param) >>
        tf.learn.Dropout("Dropout_"+i, keep_prob)
    } else {
      tf.learn.Conv2D("Conv2D_"+i, shape, stride._1, stride._2, SamePadding) >>
        tf.learn.AddBias(name = "Bias_"+i) >>
        tf.learn.ReLU("ReLU_"+i, relu_param)
    }

  /**
    * CNN architecture for GOES XRay flux (short wavelength)
    * */
  val cnn_goes_v1 = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      conv2d_unit(Shape(2, 2, 4, 64), (1, 1))(0) >>
      conv2d_unit(Shape(2, 2, 64, 32), (2, 2))(1) >>
      conv2d_unit(Shape(2, 2, 32, 16), (4, 4))(2) >>
      conv2d_unit(Shape(2, 2, 16, 8), (8, 8), dropout = false)(3) >>
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
