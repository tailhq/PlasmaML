package io.github.mandar2812.PlasmaML.vanAllen

import io.github.mandar2812.PlasmaML.SPDFData
import io.github.mandar2812.dynaml.pipes.{DataPipe, DynaMLPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils
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
    *
    * @param dataRoot The location on the disk to store the downloaded files
    *
    * */
  def download(year: Int = 1990, dataRoot: String = "data/") = {
    assert(
      Seq(1990, 1991) contains year,
      "CRRES data is available only for years 1990-1991 A.D.")

    val ftpIndexProcess = DataPipe((s: String) => s.split("\\n").toStream) >
      DynaMLPipe.replaceWhiteSpaces >
      StreamDataPipe((line: String) => line.split(",").last) >
      StreamDataPipe((fileStr: String) => fileStr.contains(".cdf")) >
      StreamDataPipe((fileStr: String) => SPDFData.nasa_spdf_baseurl + crres_uri + year+ "/" + fileStr) >
      StreamDataPipe((url: String) => {
        val filename = url.split("/").last
        logger.info("Downloading file: "+url)
        utils.downloadURL(url, dataRoot + filename)
      })

    //Since Jsoup does not support ftp protocols, we use the scala.io.Source object.
    logger.info("Getting file index from: "+SPDFData.nasa_spdf_baseurl + crres_uri + year + "/")
    ftpIndexProcess(Source.fromURL(SPDFData.nasa_spdf_baseurl + crres_uri + year + "/").mkString)
  }

}
