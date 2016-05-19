package io.github.mandar2812.PlasmaML

import java.io.File

import io.github.mandar2812.PlasmaML.cdf.{CdfContent, CdfReader}
import io.github.mandar2812.PlasmaML.cdf.util.CdfList
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 14/5/16.
  */
object PlasmaML {

  val dataDir = "data/"

  var leapSecondsFile: String = dataDir+"CDFLeapSeconds.txt"

  def convertCDF(writeToFile: String) = DataPipe((file: String) => {
    val r = new CdfContent(new CdfReader(new File(file)))
    new CdfList(r, System.out, false).run()
  })
}
