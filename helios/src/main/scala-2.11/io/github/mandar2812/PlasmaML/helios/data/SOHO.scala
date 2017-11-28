package io.github.mandar2812.PlasmaML.helios.data

import collection.JavaConverters._
import ammonite.ops._
import org.joda.time.{Interval, LocalDate, Period}
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

  object Resolutions {
    val s512 = 512
    val s1024 = 1014
  }

}

object SOHOLoader {

  import SOHO._

  /**
    * Download all the available images
    * for a given date, corresponding to
    * some specified instrument code.
    * */
  def fetch_urls(path: Path)(instrument: String, size: Int = 512)(year: Int, month: Int, day: Int) = {

    //Construct the url to download file manifest for date in question.
    val download_url = base_url+year+"/"+instrument+"/"+year+"%02d".format(month)+"%02d".format(day)+"/"

    val doc = Jsoup.connect(download_url)
      .timeout(0)
      .get()

    //Extract the elements containing the data file urls
    val elements = doc.select("a[href]")
      .iterator()
      .asScala

    val hrefs = elements.map(_.attr("href")).filter(_.contains(size+".jpg")).toList

    print("Number of files = ")
    pprint.pprintln(hrefs.length)

    println("Files identified: ")

    pprint.pprintln(hrefs)

    hrefs.map(s => download_url+s)
  }

  /**
    * Download images taken on a specified date.
    *
    * @param path The root path where the data will be downloaded,
    *             the downloader appends soho/[instrument]/[year] to
    *             the path supplied and places the images in there if
    *             the createDirTree flag is set to true.
    *
    * @param createDirTree If this is set to false, then the images
    *                      are placed directly in the path supplied.
    *
    * @param instrument The instrument code as a string, see [[SOHO.Instruments]]
    *
    * @param size The resolution of the images, defaults to 512 &times; 512
    *
    * @param date a Joda time [[LocalDate]] instance.
    * */
  def download(
    path: Path, createDirTree: Boolean = true)(
    instrument: String, size: Int = 512)(
    date: LocalDate): Unit = {
    val (year, month, day) = (date.getYear, date.getMonthOfYear, date.getDayOfMonth)

    val download_path = if(createDirTree) path/'soho/instrument/year.toString else path

    if(!(exists! download_path)) {
      mkdir! download_path
    }

    print("Downloading image manifest from the SOHO Archive for ")
    pprint.pprintln(date)


    print("Downloading images to ")
    pprint.pprintln(download_path)

    download_batch(download_path)(fetch_urls(path)(instrument, size)(year, month, day))
  }

  /**
    * Perform a bulk download of images within some date range
    * */
  def bulk_download(
    path: Path, createDirTree: Boolean = true)(
    instrument: String, size: Int = 512)(
    start: LocalDate, end: LocalDate): Unit = {

    download_range(download(path, createDirTree)(instrument, size))(start, end)
  }

}
