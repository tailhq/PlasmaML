package io.github.mandar2812.PlasmaML.vanAllen

import io.github.mandar2812.PlasmaML.SPDFData
import io.github.mandar2812.dynaml.pipes.{DataPipe, IterableDataPipe}
import io.github.mandar2812.dynaml.{DynaMLPipe, utils}
import org.apache.log4j.Logger

import scala.io.Source

/**
  * @author mandar2812 date: 30/5/16.
  *
  * Singleton object to extract CRRES data
  * from NASA SPDF in the form of CDF format
  * files.
  */
object CRRESData {

  val logger = Logger.getLogger(this.getClass)

  val crres_uri = "pub/data/crres/particle_mea/mea_h0_cdaweb/"

  /**
    * Download a year of CRRES data to the local disk.
    *
    * @param year The year: 1990 or 1991
    * @param dataRoot The location on the disk to store the downloaded files
    *
    * */
  def download(year: Int = 1990, dataRoot: String = "data/") = {
    assert(
      Seq(1990, 1991) contains year,
      "CRRES data is available only for years 1990-1991 A.D.")

    val ftpIndexProcess = DataPipe((s: String) => s.split("\\n").toStream) >
      DynaMLPipe.replaceWhiteSpaces >
      IterableDataPipe((line: String) => line.split(",").last) >
      IterableDataPipe((fileStr: String) => fileStr.contains(".cdf")) >
      IterableDataPipe((fileStr: String) => SPDFData.nasa_spdf_baseurl + crres_uri + year+ "/" + fileStr) >
      IterableDataPipe((url: String) => {
        val filename = url.split("/").last
        logger.info("Downloading file: "+filename)
        utils.downloadURL(url, dataRoot + filename)
      })

    //Since Jsoup does not support ftp protocols, we use the scala.io.Source object.
    logger.info("Getting file index from: "+SPDFData.nasa_spdf_baseurl + crres_uri + year + "/")
    ftpIndexProcess(Source.fromURL(SPDFData.nasa_spdf_baseurl + crres_uri + year + "/").mkString)
  }

  /**
    * Display help message
    *
    * */
  def help(topic: String = "about") = topic match {
    case "about" =>
      println("CRRES mission data download service: ")
      println("----------------------------------------------------------------------------")
      println("The CRRES mission was launched by NASA in 1990 to "+
        "collect information about the Earth's radiation belt. "+
        "The CRRES data is a less detailed version of the Van Allen radiation probes data sets.")
      println("Type CRRESData.help(\"usage\") for more information.")
    case "usage" =>
      println("To download: Use CRRESData.download(year, location)")
      println("Where year must be an integer (1990 or 1991) and "+
        "location specifies the fully qualified path to dump the data files into.")
  }

}
