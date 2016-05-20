package io.github.mandar2812.PlasmaML.cdf

import java.io.File

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 20/5/16.
  */
object CDFUtils {

  val dataDir = "data/"

  var leapSecondsFile: String = dataDir+"CDFLeapSeconds.txt"

  def getVariableAttributes(cdf: CdfContent) =
    cdf.getVariables.map(v => (
      v.getName,
      cdf.getVariableAttributes.map(attr =>
        (attr.getName, attr.getEntry(v) match {
            case null => ""
            case obj => obj.toString
          })
      ).toMap)
    ).toMap


  def cdfContentAndAttributes = DataPipe((file: String) => {
    val content = new CdfContent(new CdfReader(new File(file)))
    (content, getVariableAttributes(content))
  })

  def cdfToRDD() = {

  }
}