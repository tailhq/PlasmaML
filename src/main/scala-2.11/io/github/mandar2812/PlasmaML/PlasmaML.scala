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
  def main(args0: Array[String]): Unit = {
    DynaML.main0(args0.toList, System.in, System.out, System.err) match{
      case Left((success, msg)) =>
        if (success) {
          Console.out.println(msg)
          sys.exit(0)
        } else {
          Console.err.println(msg)
          sys.exit(1)
        }
      case Right(success) =>
        if (success) sys.exit(0)
        else sys.exit(1)
    }
  }
}
