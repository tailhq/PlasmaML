package io.github.mandar2812.PlasmaML.vanAllen

import scala.collection.mutable.{MutableList => ML}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date, GregorianCalendar}
import java.nio.file.{Files, Paths}

import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger
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

  def combineFileStreams(streams: Stream[String]*) =
    streams.reduceLeft[Stream[String]]((coup1, coup2) => coup1 ++ coup2)

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

    val greg: GregorianCalendar = new GregorianCalendar()
    val calendar = Calendar.getInstance()

    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd")

    val dateS: Date = sdf.parse(startDate)
    val dateE: Date = sdf.parse(endDate)

    calendar.setTime(dateE)
    greg.setTime(dateS)

    var columnData: Option[JValue] = None
    val content = ML[Stream[String]]()


    while(greg.before(calendar)) {
      val year = greg.get(Calendar.YEAR)
      val doy = greg.get(Calendar.DAY_OF_YEAR)

      probes.foreach(probe => {
        val fileContent = fileToStreamOption.run(category+"_"+probe+"_"+year+"_"+doy+".txt")
        fileContent match {
          case Some(cont) =>
            val (buff, json) = stripFileHeader().run(cont)
            if(columnData.isEmpty & json.isDefined) {
              columnData = json
            }
            content += buff

          case None =>
            logger.info("No file found: "+category+" probe "+probe+" for "+year+"/"+doy)
        }
      })
      greg.add(Calendar.DAY_OF_YEAR, 1)
    }

    combineFileStreams(content:_*)
  }
}
