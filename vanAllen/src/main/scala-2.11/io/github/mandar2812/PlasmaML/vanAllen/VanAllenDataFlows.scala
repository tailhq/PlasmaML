package io.github.mandar2812.PlasmaML.vanAllen

import java.nio.file.{Files, Paths}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date, GregorianCalendar}

import scala.collection.mutable.{MutableList => ML}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
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

  val logger = Logger.getLogger(this.getClass)

  var dataRoot: String = "data/"

  val headerLead: Char = '#'

  val headerEndStr:String = " End JSON"

  val cleanRegex = """\}(\"\w)""".r

  val sparkHost = "local[*]"

  private val greg = new GregorianCalendar()

  val sc = new SparkContext(new SparkConf().setMaster(sparkHost).setAppName("Van Allen Data Models"))

  /**
    * Returns a DynaML [[DataPipe]] that takes a [[Stream]] of lines
    * and strips the header json from them.
    *
    * In case the header is required, strip it of undesirable
    * leading and trailing characters and return a parsable json string
    * */
  def stripFileHeader(keepHeader: Boolean = false) =
    DataPipe((stream: Stream[String]) => {
      val (content, header) = stream.partition(_.head != headerLead)
      val processedHeader = header.map(line =>
        line.takeRight(line.length-2)
          .trim
          .replace(headerEndStr, ""))

      keepHeader match {
        case false =>
          (content, None)
        case true =>
          val jsonParsed = try {
            Some(parse(cleanRegex.replaceAllIn(processedHeader.mkString(""), """\}\,$1"""), false))
          } catch {
            case e: Exception => None
          }

          (content, jsonParsed)
      }
    })

  def fileToStreamOption = DataPipe((relativePath: String) => {
    Files.exists(Paths.get(dataRoot+relativePath)) match {
      case false => None
      case true => Some(utils.textFileToStream(dataRoot+relativePath))
    }
  })

  def fileToRDDOption = DataPipe((relativePath: String) => {
    Files.exists(Paths.get(dataRoot+relativePath)) match {
      case false => None
      case true => Some(sc.textFile(dataRoot+relativePath))
    }
  })

  def processDateString(d: String, sep: Char = '/'): (Int, Int, Int) = {
    val spl = d.split(sep).map(_.toInt)
    (spl.head, spl(1), spl(2))
  }

  def dateRange(start: String,
                end: String,
                format: String = "yyyy/MM/dd"): Stream[Date] = {

    val dates = ML[Date]()

    val sdf = new SimpleDateFormat("yyyy/MM/dd")

    val dateS = sdf.parse(start)
    val dateE = sdf.parse(end)
    val calendar = Calendar.getInstance()
    calendar.setTime(dateE)

    greg.setTime(dateS)

    while(greg.before(calendar)) {
      dates += greg.getTime
      greg.add(Calendar.DAY_OF_YEAR, 1)
    }
    dates.toStream
  }

  /**
    * Extract data from a particular category
    * within a specified date range
    *
    * */
  def extractData(startDate: String,
                  endDate: String,
                  category: String = VanAllenData.dataCategories.keys.head,
                  probes: Seq[String] = VanAllenData.probes,
                  columns: Seq[String] = Seq()) = {


    var columnData: Option[JValue] = None

    val dates = dateRange(startDate, endDate)

    probes.map(probe => {
      val data = dates.map(date => {
        greg.setTime(date)
        val doy = greg.get(Calendar.DAY_OF_YEAR)
        val year = greg.get(Calendar.YEAR)

        val fileContent = fileToStreamOption.run(category + "_" + probe + "_" + year + "_" + doy + ".txt")
        fileContent match {
          case Some(cont) =>
            val (buff, json) = stripFileHeader(true).run(cont)
            if (columnData.isEmpty & json.isDefined) {
              columnData = json
            }
            buff

          case None =>
            logger.info("No file found: " + category + " probe " + probe + " for " + year + "/" + doy)
            Stream()
        }
      }).reduceLeft[Stream[String]](_++_)
      (probe, data)
    }).toMap
  }
}
