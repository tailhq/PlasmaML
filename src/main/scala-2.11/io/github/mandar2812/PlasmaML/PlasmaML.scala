package io.github.mandar2812.PlasmaML

import io.github.mandar2812.dynaml.DynaML


/**
  * Created by mandar on 16/06/2017.
  */
object PlasmaML {

  /**
    * The command-line entry point, which does all the argument parsing before
    * delegating to [[DynaML.run]]
    */
  def main(args0: Array[String]): Unit = DynaML.main(args0)

}
