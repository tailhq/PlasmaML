package io.github.mandar2812.PlasmaML.vanAllen

import java.text.SimpleDateFormat
import java.util.{Calendar, Date, GregorianCalendar}

import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger
import org.jsoup.Jsoup

import scala.collection.mutable.{MutableList => ML}
import collection.JavaConverters._

/**
  * @author mandar 4/5/16.
  *
  * Utility to download and pre-process van allen data sets
  * from the John's Hopkins Applied Physics laboratory web pages.
  */
object VanAllenData {

  val logger = Logger.getLogger(this.getClass)

  val dataCategories: Map[String, Map[String, String]] = Map(

    "position" -> Map(
      "abstract" -> "Position of probes in GSE, GSM and L-shell coordinate systems",

      "summary" -> (
        "RBSPICE will determine how space weather creates what is called the “storm-time ring current”"+
        " around Earth and determine how that ring current supplies and "+
        "supports the creation of radiation populations.")),

    "HOPE" -> Map("abstract" -> "Helium Oxygen Proton Electron",
      "summary" -> ("Helium Oxygen Proton Electron :\n"+
        "HOPE uses an electrostatic top-hat analyzer and time-gated coincidence detectors to measure electrons, "+
          "protons, and helium and oxygen ions with energies from less than or equal to 20 eV or spacecraft potential"+
          " (whichever is greater) to greater than or equal to 45 keV while rejecting penetrating backgrounds.")),


    "EMFISIS" -> Map(
      "abstract" -> "Electric and Magnetic Field Instrument Suite and Integrated Science",

      "summary" -> (
        "The EMFISIS investigation will focus on the important role played by magnetic fields and "+
        "plasma waves in the processes of radiation belt particle acceleration and loss. EMFISIS offers "+
        "the opportunity to understand the origin of important magnetospheric plasma waves as well as the "+
        "evolution of the magnetic field that defines the basic coordinate system controlling the structure of "+
        "the radiation belts and the storm-time ring current.")),

    "RBSPICE" -> Map(
      "abstract" -> "Radiation Belt Storm Probes Ion Composition Experiment",

      "summary" -> (
        "RBSPICE will determine how space weather creates what is called the “storm-time ring current” "+
        "around Earth and determine how that ring current supplies "+
        "and supports the creation of radiation populations.\nThis investigation will accurately measure the "+
        "ring current pressure distribution, which is needed to understand how the inner magnetosphere changes "+
        "during geomagnetic storms and how that storm environment supplies and supports the acceleration and "+
        "loss processes involved in creating and sustaining hazardous radiation particle populations.")),

    "RPS" -> Map(
      "abstract" -> "Relativistic Proton Spectrometer",

      "summary" -> (
        "The RPS will measure inner Van Allen belt protons with energies from 50 MeV to 2 GeV. "+
        "Presently, the intensity of trapped protons with energies beyond about 150 MeV is not well known "+
        "and thought to be underestimated in existing specification models. Such protons are known to pose "+
        "a number of hazards to astronauts and spacecraft, including total ionizing dose, displacement damage, "+
        "single event effects, and nuclear activation. This instrument will address a priority highly ranked by "+
        "the scientific and technical community and will extend the measurement capability of this mission to a "+
        "range beyond that originally planned."))
  )

  val dataCategorySpecs = Map(
    "LShell_GSM_GSE" -> "position",
    "SW_ECTHOPE_RBSP" -> "HOPE",
    "SW_EMFISIS_RBSP" -> "EMFISIS",
    "SW_RBSPICE_RBSP" -> "RBSPICE",
    "SW_RPS_RBSP" -> "RPS")

  val probes = Seq("A", "B")

  val jhuapl_baseurl = "http://rbspgway.jhuapl.edu/"

  val sw_data_uri = "sw_data_browser"

  /**
    * Download text data files for a given day
    *
    * @param year Year number ex: 2013
    *
    * @param doy Day of year as integer (1-365)
    *
    * @param dataRoot Directory in which the files will be downloaded
    *
    * */
  def download(year: Int = 2016,
               doy: Int = 112,
               dataRoot: String = "data/",
               categories: Seq[String] = dataCategories.keys.toSeq,
               probesSelected: Seq[String] = probes) = {

    val categoryBuffer = dataCategories.map(couple =>
      (couple._1, probes.map(p => (p, ML[String]() )).toMap))

    //Retrieve the relevant web page corresponding to the
    //given year and date
    val doc = Jsoup.connect(jhuapl_baseurl+sw_data_uri)
      .data("Year", year.toString)
      .data("Doy", "%03d".format(doy))
      .timeout(0)
      .post()

    //Extract the elements containing the data file urls
    val elements = doc.select("table[style]").select("a[href]").iterator().asScala
    val hrefs = elements.map(_.attr("href")).filterNot(_.contains(".cdf"))

    hrefs.foreach(link => {
      dataCategorySpecs.foreach(categorySpec =>
        if(link.contains(categorySpec._1+"A")) {
          categoryBuffer(categorySpec._2)("A") += link
        } else if(link.contains(categorySpec._1+"B")) {
          categoryBuffer(categorySpec._2)("B") += link
        }
      )
    })

    logger.info("Downloading files for year: "+year+" day of year: "+doy+".")

    categoryBuffer.filterKeys(categories.contains(_))
      .foreach(pair => {
      //for each category download
      logger.info(pair._1+": ")
      pair._2.filterKeys(probesSelected.contains(_))
        .foreach(probe => {
        logger.info("Probe "+probe._1)
        probe._2.foreach(hypLink =>
          utils.downloadURL(jhuapl_baseurl+hypLink,
            dataRoot+pair._1+"_"+probe._1+"_"+year+"_"+doy+".txt"))
      })
    })
  }

  def bulkDownload(start: String = "2012/01/01",
                   end: String = "2015/12/31",
                   dataRoot: String = "data/",
                   categories: Seq[String] = dataCategories.keys.toSeq,
                   probesSelected: Seq[String] = probes): Unit = {
    val greg: GregorianCalendar = new GregorianCalendar()
    val calendar = Calendar.getInstance()

    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd")

    val dateS: Date = sdf.parse(start)
    val dateE: Date = sdf.parse(end)

    calendar.setTime(dateE)
    greg.setTime(dateS)

    while(greg.before(calendar)) {

      logger.info("Date: "+greg.getTime)
      download(greg.get(Calendar.YEAR),
        greg.get(Calendar.DAY_OF_YEAR),
        dataRoot, categories,
        probesSelected)

      greg.add(Calendar.DAY_OF_YEAR, 1)
    }
  }

  /**
    *  Display help messages
    *
    *  @param topic Select type of help message
    *               help = "about"   -- general introduction
    *               help = "data"    -- information about data file categories
    *               help = category  -- summary of data category, instrument details etc
    *
    * */
  def help(topic: String = "about"): Unit = topic match {
    case "about" =>
      println("Van Allen probes data pre-processing service")
      println("\nType VanAllenData.help(\"data\") for a summary of the Van Allen probes data set or "+
        "\nVanAllenData.help(\"categories\") for an explaination of the various types of data files")
    case "data" =>
      println("The Van Allen Probes mission (formerly known as the Radiation Belt Storm Probes mission, "+
        "renamed on Nov. 9, 2012) is part of NASA’s Living With a Star geo-space program to explore "+
        "fundamental processes that operate throughout the solar system, in particular those that generate "+
        "hazardous space weather effects near the Earth and phenomena that could affect solar system exploration.")
      println("The Van Allen probes A and B orbit around the Earth's "+
        "radiation belt and record important plasma characteristics.")
    case "categories" =>
      println("Readings on the Van Allen probes are collected by a variety of instruments: ")
      println("----------------------------------------------------------------------------")
      dataCategories.foreach(cat => {
        println(cat._1+": "+cat._2("abstract"))
      })
    case _ =>
      if(dataCategories.contains(topic)) println(dataCategories(topic)("summary"))
  }
}
