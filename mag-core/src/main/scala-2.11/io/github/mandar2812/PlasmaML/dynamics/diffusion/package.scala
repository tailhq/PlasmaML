package io.github.mandar2812.PlasmaML.dynamics

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.DataPipe2

/**
  * Primitives for diffusion dynamics.
  *
  * @author mandar2812 date 10/07/2017.
  * */
package object diffusion {

  type breezeVec = DenseVector[Double]

  val sqNormBDV = DataPipe2((x: breezeVec, y: breezeVec) => {
    val d = x-y
    d dot d
  })

  val sqNormDouble = DataPipe2((x: Double, y: Double) => {
    math.pow(x-y, 2.0)
  })

  val l1NormDouble = DataPipe2((x: Double, y: Double) => {
    math.abs(x-y)
  })

  def gradSqNormDouble(order_x: Int, order_y: Int)(x: Double, y: Double): Double = {
    require(
      order_x >= 0 && order_y >= 0,
      "Orders of differentiation must be non negative!!")

    val order = order_x + order_y
    if(order > 2) 0d
    else if (order == 0) sqNormDouble(x,y)
    else (order_x, order_y) match {
      case (1, 0) => 2d*math.abs(x-y)
      case (0, 1) => -2d*math.abs(x-y)
      case (1, 1) => -2d
      case _ => 2d
    }
  }

  def gradL1NormDouble(order_x: Int, order_y: Int)(x: Double, y: Double): Double = {
    require(
      order_x >= 0 && order_y >= 0,
      "Orders of differentiation must be non negative!!")

    val order = order_x + order_y
    if(order > 1) 0d
    else if (order == 0) l1NormDouble(x,y)
    else (order_x, order_y) match {
      case (1, 0) => 1d
      case (0, 1) => 1d
      case _ => 1d
    }
  }

}
