package io.github.mandar2812.PlasmaML

/**
  * Created by mandar on 30/03/2017.
  */
abstract class RadialDiffusionSolver(
  lShellLimits: (Double, Double), timeLimits: (Double, Double),
  nL: Int, nT: Int) {

  val (deltaL, deltaT) = ((lShellLimits._2 - lShellLimits._1)/nL, (timeLimits._2 - timeLimits._1)/nT)



}
