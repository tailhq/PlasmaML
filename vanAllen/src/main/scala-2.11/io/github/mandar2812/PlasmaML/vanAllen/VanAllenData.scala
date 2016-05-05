package io.github.mandar2812.PlasmaML.vanAllen

import java.text.SimpleDateFormat
import java.util.GregorianCalendar

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

  val dataCategories = Map(
    "position" -> Map("description" ->
      "Position of probes in GSE, GSM and L-shell coordinate systems"),
    "HOPE" -> Map("description" -> ""),
    "EMFISIS" -> Map("description" ->
      "Electric and Magnetic Field Instrument Suite and Integrated Science"),
    "RBSPICE" -> Map("description" ->
      "Radiation Belt Storm Probes Ion Composition Experiment"),
    "REPT" -> Map("description" ->
      ""),
    "RPS" -> Map("description" -> "Relativistic Proton Spectrometer"))

  val dataCategorySpecs = Map(
    "LShell_GSM_GSE" -> "position",
    "SW_ECTHOPE_RBSP" -> "HOPE",
    "SW_EMFISIS_RBSP" -> "EMFISIS",
    "SW_RBSPICE_RBSP" -> "RBSPICE",
    "SW_ECTREPT_RBSP" -> "REPT",
    "SW_RPS_RBSP" -> "RPS")

  val probes = Seq("A", "B")

  val jhuapl_baseurl = "http://rbspgway.jhuapl.edu/"

  val sw_data_uri = "sw_data_browser"

  def createDirectoryTree(baseDir: String): Unit = {

  }

  def download(year: Int = 2016, doy: Int = 112) = {

    val categoryBuffer = dataCategories.map(couple => (couple._1, ML[String]()))

    //Retrieve the relevant web page corresponding to the
    //given year and date
    val doc = Jsoup.connect(jhuapl_baseurl+sw_data_uri)
      .data("Year", year.toString)
      .data("Doy", doy.toString)
      .post()

    //Extract the elements containing the data file urls
    val elements = doc.select("table[style]").select("a[href]").iterator().asScala
    val hrefs = elements.map(_.attr("href")).filterNot(_.contains(".cdf"))
    hrefs.foreach(link => {
      //logger.info(link)
      dataCategorySpecs.foreach(categorySpec =>
        if(link.contains(categorySpec._1)) categoryBuffer(categorySpec._2) += link
      )
    })
    logger.info(categoryBuffer)
  }

  def bulkDownload(start: String = "2012/01/01",
                   end: String = "2015/12/31"): Unit = {
    val greg: GregorianCalendar = new GregorianCalendar()
    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd")

  }

  def help(topic: String = "about"): Unit = topic match {
    case "about" =>
      println("Van Allen probes data pre-processing service")
    case "data" =>
      println("The Van Allen probes A and B orbit around the Earth's "+
        "radiation belt and record important plasma parameters.")
    case "categories" =>
      println("Readings on the Van Allen probes are collected by a variety of instruments: ")
      println("----------------------------------------------------------------------------")
      dataCategories.foreach(cat => {
        println(cat._1+": "+cat._2("description"))
      })
  }
}
