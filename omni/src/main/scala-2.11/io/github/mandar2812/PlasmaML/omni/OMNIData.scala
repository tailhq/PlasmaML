package io.github.mandar2812.PlasmaML.omni

import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamFlatMapPipe}
import org.apache.log4j.Logger
import org.joda.time.{DateTime, DateTimeZone}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

/**
  *
  *
  * @author mandar2812 date 12/7/16.
  * */
object OMNIData {

  val logger = Logger.getLogger(this.getClass)

  val omni_uri = "pub/data/omni/"

  object Quantities {

    //Sunspot Number
    val sunspot_number = 39

    //Genmagnetic Indices
    val Dst = 40
    val AE = 41
    val Kp = 38

    //L1 quantities
    val V_SW = 24
    val B_Z = 16
    val P = 28
  }

  /**
    * Stores the missing value strings for
    * each column index in the hourly resolution
    * OMNI files.
    * */
  var columnFillValues = Map(
    16 -> "999.9", 21 -> "999.9",
    24 -> "9999.", 23 -> "999.9",
    40 -> "99999", 22 -> "9999999.",
    25 -> "999.9", 28 -> "99.99",
    27 -> "9.999", 39 -> "999",
    45 -> "99999.99", 46 -> "99999.99",
    47 -> "99999.99", 15 -> "999.9")

  /**
    * Contains the name of the quantity stored
    * by its column index (which starts from 0).
    * */
  var columnNames = Map(
    24 -> "Solar Wind Speed",
    16 -> "I.M.F Bz",
    40 -> "Dst",
    41 -> "AE",
    38 -> "Kp",
    39 -> "Sunspot Number",
    28 -> "Plasma Flow Pressure")

  //The column indices corresponding to the year, day of year and hour respectively
  val dateColumns = List(0, 1, 2)

  def getFilePattern(year: Int): String = "omni2_"+year+".csv"

}

object OMNILoader {

  import io.github.mandar2812.PlasmaML.omni.OMNIData.{columnFillValues, dateColumns}

  //Set time zone to UTC.
  DateTimeZone.setDefault(DateTimeZone.UTC)

  val dt_format: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/D/H")

  /**
    * Returns a [[io.github.mandar2812.dynaml.pipes.DataPipe]] which
    * reads an OMNI file cleans it and extracts the columns specified
    * by targetColumn and exogenousInputs.
    * */
  def omniFileToStream(targetColumn: Int, exogenousInputs: List[Int]) =
    fileToStream >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      dateColumns++List(targetColumn)++exogenousInputs,
      columnFillValues) >
    removeMissingLines


  val processWithDateTime = DataPipe((lines: Stream[String]) => lines.map{line =>
    val splits = line.split(",")

    val dt_string = splits.take(3).mkString("/")
    val data = splits.last.toDouble
    val timestamp = DateTime.parse(dt_string, dt_format)
    (timestamp, data)
  })


  val forward_time_window = (deltaT: Int, timelag: Int) =>
    DataPipe((lines: Stream[(DateTime, Double)]) =>
      lines.toList.sliding(deltaT+timelag+1).map((history) => {

        val features: Seq[Double] = history.slice(deltaT, deltaT+timelag).map(_._2)

        (history.head._1, features)
      }).toStream)

  def omniDataToSlidingTS(deltaT: Int, timelag: Int)(targetColumn: Int = OMNIData.Quantities.V_SW) =
    StreamFlatMapPipe(omniFileToStream(targetColumn, List())) >
      processWithDateTime >
      forward_time_window(deltaT, timelag)
}
