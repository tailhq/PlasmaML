package io.github.mandar2812

import java.io.File

import io.github.mandar2812.PlasmaML.cdf.{CdfContent, CdfReader}
import io.github.mandar2812.PlasmaML.cdf.util.CdfList
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 14/5/16.
  */
package object PlasmaML {
  def convertCDF(writeToFile: String) = DataPipe((file: String) => {
    val r = new CdfContent(new CdfReader(new File(file)))
    new CdfList(r, System.out, false).run()
  })
}
