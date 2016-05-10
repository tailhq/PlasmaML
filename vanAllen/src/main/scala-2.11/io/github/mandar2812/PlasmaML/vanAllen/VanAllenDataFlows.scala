package io.github.mandar2812.PlasmaML.vanAllen

import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import org.json4s._
import org.json4s.jackson.JsonMethods._

/**
  * @author mandar on 10/5/16.
  *
  * A utility object that contains many
  * reusable data workflows that are required
  * for pre-processing of the van allen probe
  * data files.
  */
object VanAllenDataFlows {

  /**
    * Returns a DynaML [[DataPipe]] that takes a [[Stream]] of lines
    * and strips the header json from them.
    * */
  def stripFileHeader() = StreamDataPipe((line: String) => line.head != '#')

  /**
    * In case the header is required, strip it of undesirable
    * leading and trailing characters and return a parsable json string
    * */
  def stripFileHeaderWithJSON() = DataPipe((stream: Stream[String]) => {
        val (content, header) = stream.partition(_.head != '#')
        val processedHeader = header.map(line =>
          line.takeRight(line.length-2)
            .trim
            .replace(" End JSON", ""))
        val cleanRegex = """\}(\"\w)""".r
        val jsonParsed = parse(cleanRegex.replaceAllIn(processedHeader.mkString(""), """\}\,$1"""), false)
        (content, jsonParsed)
      })

}
