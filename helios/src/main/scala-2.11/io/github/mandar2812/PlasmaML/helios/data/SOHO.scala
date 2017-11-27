package io.github.mandar2812.PlasmaML.helios.data

import collection.JavaConverters._
import ammonite.ops._
import io.github.mandar2812.dynaml.utils
import org.joda.time.LocalDate
import org.jsoup.Jsoup

/**
  * Helper class for downloading solar images from the
  * <a href="https://sohowww.nascom.nasa.gov">SOHO</a> archive.
  * @author mandar2812 date 27/11/2017
  * */
object SOHO {

  /**
    * The url for FTP download
    * */
  val base_url = "https://sohowww.nascom.nasa.gov/data/REPROCESSING/Completed/"

  /**
    * Instrument codes for the SOHO satellite
    * */
  object Instruments {

    val MDIMAG = "mdimag"

    val MDIIGR = "mdiigr"

    val EIT171 = "eit171"

    val EIT195 = "eit195"

    val EIT284 = "eit284"

    val EIT304 = "eit304"
  }

}

object SOHOLoader {

  import SOHO._

  /**
    * Download a resource (image, file) from a sequence of urls to a specified
    * disk location.
    * */
  def download_batch(path: Path)(urls: List[String]): Unit = {
    urls.par.foreach(s => utils.downloadURL(s, (path/s.split('/').last).toString()))
  }

  /**
    * Download all the available images
    * for a given date, corresponding to
    * some specified instrument code.
    * */
  def fetch_urls(path: Path)(instrument: String, size: Int = 512)(year: Int, month: Int, day: Int) = {

    //Construct the url to download file manifest for date in question.
    val download_url = base_url+year+"/"+instrument+"/"+year+"%02d".format(month)+day+"/"

    val doc = Jsoup.connect(download_url)
      .timeout(0)
      .get()

    //Extract the elements containing the data file urls
    val elements = doc.select("a[href]")
      .iterator()
      .asScala

    val hrefs = elements.map(_.attr("href")).filter(_.contains(size+".jpg")).toList

    println("Files identified: ")

    hrefs.foreach(println(_))

    hrefs.map(s => download_url+s)
  }

  def download(path: Path)(instrument: String, size: Int = 512)(date: LocalDate): Unit = {
    val (year, month, day) = (date.getYear, date.getMonthOfYear, date.getDayOfMonth)

    val download_path = path/instrument/year.toString

    if(!(exists! download_path)) {
      mkdir! download_path
    }

    print("Downloading image manifest from the SOHO Archive for ")
    pprint.pprintln(date)


    print("Downloading images to ")
    pprint.pprintln(download_path)

    download_batch(download_path)(fetch_urls(path)(instrument, size)(year, month, day))
  }

}
