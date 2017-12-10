package io.github.mandar2812.PlasmaML.helios.data

import collection.JavaConverters._
import ammonite.ops._
import org.joda.time._
import org.jsoup.Jsoup

import scala.util.matching.Regex

/**
  * Helper object for downloading solar images from the
  * <a href="https://sohowww.nascom.nasa.gov">SOHO</a> archive.
  * @author mandar2812 date 27/11/2017
  * */
object SOHOData {

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

  def getFilePattern(date: LocalDate, source: SOHO): Regex = {
    val (year, month, day) = (date.getYear.toString, date.getMonthOfYear.toString, date.dayOfMonth.toString)

    (year+month+day+"""_(\d{4}?)_"""+source.instrument+"_"+source.size+"""\.jpg""").r
  }

  def getFilePattern(date: YearMonth, source: SOHO): Regex = {
    val (year, month) = (date.getYear.toString, date.getMonthOfYear.toString)

    (year+month+"""(\d{2}?)_(\d{4}?)_"""+source.instrument+"_"+source.size+".jpg").r
  }

}

object SOHOLoader {

  import SOHOData._

  DateTimeZone.setDefault(DateTimeZone.UTC)

  /**
    * Download all the available images
    * for a given date, corresponding to
    * some specified instrument code.
    * */
  def fetch_urls(path: Path)(instrument: String, size: Int = 512)(year: Int, month: Int, day: Int) = {

    //Construct the url to download file manifest for date in question.
    val download_url = base_url+year+"/"+instrument+"/"+year+"%02d".format(month)+"%02d".format(day)+"/"

    val hrefs = try {
      val doc = Jsoup.connect(download_url).timeout(0).get()
      //Extract the elements containing the data file urls
      val elements = doc.select("a[href]")
        .iterator()
        .asScala

      elements.map(_.attr("href")).filter(_.contains(size+".jpg")).toList
    } catch {
      case _: org.jsoup.HttpStatusException =>
        println("Not found: "+download_url)
        List.empty[String]
      case _: Exception => List.empty[String]
    }

    println("Number of files = "+hrefs.length)

    hrefs.map(s => download_url+s)
  }

  /**
    * Download images taken on a specified date.
    *
    * @param path The root path where the data will be downloaded,
    *             the downloader appends soho/[instrument]/[year] to
    *             the path supplied and places the images in there if
    *             the createDirTree flag is set to true.
    * @param createDirTree If this is set to false, then the images
    *                      are placed directly in the path supplied.
    * @param instrument The instrument code as a string, see [[SOHOData.Instruments]]
    * @param size The resolution of the images, defaults to 512 &times; 512
    * @param date a Joda time [[LocalDate]] instance.
    * */
  def download(
    path: Path, createDirTree: Boolean = true)(
    instrument: String, size: Int = 512)(
    date: LocalDate): Unit = {
    val (year, month, day) = (date.getYear, date.getMonthOfYear, date.getDayOfMonth)

    val download_path = if(createDirTree) path/'soho/instrument/year.toString/"%02d".format(month) else path

    if(!(exists! download_path)) {
      mkdir! download_path
    }

    println("Downloading image manifest from the SOHO Archive for: "+date+"\nDownload Path: "+download_path)

    download_batch(download_path)(fetch_urls(path)(instrument, size)(year, month, day))
  }

  /**
    * Perform a bulk download of images within some date range
    * */
  def bulk_download(
    path: Path, createDirTree: Boolean = true)(
    instrument: String, size: Int = 512)(
    start: LocalDate, end: LocalDate): Unit = {

    download_day_range(download(path, createDirTree)(instrument, size))(start, end)
  }


  /**
    * Load paths to SOHO images from the disk.
    *
    * @param soho_files_path The base directory in which soho data is stored.
    *
    * @param year_month The year-month from which the images were taken.
    *
    * @param soho_source A data source i.e. a [[SOHO]] instance.
    *
    * @param dirTreeCreated If SOHO directory structure has been
    *                       created inside the soho_files_path,
    *                       defaults to true.
    *
    * @return Time stamped images for some soho instrument and image resolution.
    * */
  def load_images(
    soho_files_path: Path, year_month: YearMonth,
    soho_source: SOHO, dirTreeCreated: Boolean = true): Stream[(DateTime, Path)] = {


    val (year, month) = (year_month.getYear.toString, year_month.getMonthOfYear.toString)
    val filePattern = getFilePattern(year_month, soho_source)

    val image_paths = if(dirTreeCreated) {
      ls! soho_files_path |? (s => s.isDir && s.segments.last == soho_source.instrument) ||
        (d => {
          ls! d |?
            (s => s.isDir && s.segments.contains(year)) ||
            (ls! _) |?
            (_.segments.contains(month))
        }) ||
        (ls! _ )
    } else {
      ls.rec! soho_files_path
    }

    (image_paths | (file => {
      (filePattern.findFirstMatchIn(file.segments.last), file)
    }) |? (_._1.isDefined) | (c => {
      val Some(matchStr) = c._1

      val (day, time) = (matchStr.group(1), matchStr.group(2))

      (
        new DateTime(
          year.toInt, month.toInt, day.toInt,
          time.take(2).toInt, time.takeRight(2).toInt),
        c._2
      )
    })).toStream
  }

}
