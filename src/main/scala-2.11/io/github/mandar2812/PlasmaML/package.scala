package io.github.mandar2812

import java.io.File

import uk.ac.bristol.star.cdf.{CdfContent, CdfReader}

/**
  * Created by mandar on 14/5/16.
  */
package object PlasmaML {
  def convertCDF(path: String) = {
    val r = new CdfContent(new CdfReader(new File(path)))
    r
  }
}
