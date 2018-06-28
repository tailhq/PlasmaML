package io.github.mandar2812.PlasmaML.helios.data

import collection.JavaConverters._
import ammonite.ops._
import org.joda.time.{DateTime, DateTimeZone, LocalDate, YearMonth}
import org.jsoup.Jsoup

import scala.util.matching.Regex

/**
  * Helper class for downloading solar images from the
  * <a href="https://sdo.gsfc.nasa.gov">Solar Dynamics Observatory</a> archive.
  * @author mandar2812 date 27/11/2017
  * */
object SDOData {

  /**
    * The url for FTP download
    * */
  val base_url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"

  /**
    * Instrument codes for the SDO satellite
    * */
  object Instruments {

    //HMI images: Helioseismic & Magnetic Imager
    val HMIIC = "HMIIC"

    val HMIIF = "HMIIF"

    val HMID = "HMID"

    val HMIB = "HMIB"

    //AIA images: Atmospheric Imaging Assembly
    val AIA171 = "0171"

    val AIA131 = "0131"

    val AIA193 = "0193"

    val AIA211 = "0211"

    val AIA1600 = "1600"

    val AIA94 = "0094"

    val AIA335 = "0335"

    val AIA304 = "0304"

    //Composite images
    val HMI171 = "HMI171"

    val AIA094335193 = "094335193"
  }

  object Resolutions {
    val s512 = 512
    val s1024 = 1014
    val s2048 = 2048
    val s4096 = 4096
  }

  def getFilePattern(date: LocalDate, source: SDO): Regex = {
    val (year, month, day) = (
      date.getYear.toString,
      "%02d".format(date.getMonthOfYear),
      "%02d".format(date.getDayOfMonth))

    (year+month+day+"""_(\d{6}?)_"""+"_"+source.size+"_"+source.instrument+"\\.jpg").r
  }

  def getFilePattern(date: YearMonth, source: SDO): Regex = {
    val (year, month) = (date.getYear.toString, "%02d".format(date.getMonthOfYear))

    (year+month+"""(\d{2}?)_(\d{6}?)_"""+source.size+"_"+source.instrument+"\\.jpg").r
  }

  def getFilePattern(date: YearMonth, sources: Seq[SDO]): Regex = {
    val (year, month) = (
      date.getYear.toString,
      "%02d".format(date.getMonthOfYear))

    (year+month+"""(\d{2}?)_(\d{6}?)_("""+sources.map(s => s.size+"_"+s.instrument).mkString("|")+")\\.jpg").r
  }


}


object SDOLoader {

  import SDOData._

  DateTimeZone.setDefault(DateTimeZone.UTC)

  /**
    * Download all the available images
    * for a given date, corresponding to
    * some specified instrument code.
    * */
  def fetch_urls(path: Path)(instrument: String, size: Int = 512)(year: Int, month: Int, day: Int) = {

    //Construct the url to download file manifest for date in question.
    val download_url = base_url+year+"/"+"%02d".format(month)+"/"+"%02d".format(day)+"/"

    val hrefs = try {
      val doc = Jsoup.connect(download_url).timeout(0).get()
      //Extract the elements containing the data file urls
      val elements = doc.select("a[href]")
        .iterator()
        .asScala

      elements.map(_.attr("href")).filter(_.contains("_"+size+"_"+instrument+".jpg")).toList
    } catch {
      case _: org.jsoup.HttpStatusException => List.empty[String]
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
    * @param instrument The instrument code as a string, see [[SDOData.Instruments]]
    * @param size The resolution of the images, defaults to 512 &times; 512
    * @param date a Joda time [[LocalDate]] instance.
    * */
  def download(
    path: Path, createDirTree: Boolean = true)(
    instrument: String, size: Int = 512)(
    date: LocalDate): Unit = {
    val (year, month, day) = (date.getYear, date.getMonthOfYear, date.getDayOfMonth)

    val download_path = if(createDirTree) path/'sdo/instrument/year.toString/"%02d".format(month) else path

    if(!(exists! download_path)) {
      mkdir! download_path
    }

    println("Downloading image manifest from the SDO Archive for: "+date+"\nDownload Path: "+download_path)

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
    * Load paths to [[SDO]] images from the disk.
    *
    * @param sdo_files_path The base directory in which sdo data is stored.
    *
    * @param year_month The year-month from which the images were taken.
    *
    * @param sdo_source A data source i.e. a [[SDO]] instance.
    *
    * @param dirTreeCreated If SDO directory structure has been
    *                       created inside the soho_files_path,
    *                       defaults to true.
    *
    * @return Time stamped images for some soho instrument and image resolution.
    * */
  def load_images(
    sdo_files_path: Path, year_month: YearMonth,
    sdo_source: SDO, dirTreeCreated: Boolean = true): Iterable[(DateTime, Path)] = {


    val (year, month) = (year_month.getYear.toString, "%02d".format(year_month.getMonthOfYear))
    val filePattern = getFilePattern(year_month, sdo_source)

    val image_paths = if(dirTreeCreated) {
      ls! sdo_files_path |? (s => s.isDir && s.segments.last == sdo_source.instrument) ||
        (d => {
          ls! d |?
            (s => s.isDir && s.segments.contains(year)) ||
            (ls! _) |?
            (_.segments.contains(month))
        }) ||
        (ls! _ )
    } else {
      ls.rec! sdo_files_path
    }

    (image_paths | (file => {
      (filePattern.findFirstMatchIn(file.segments.last), file)
    }) |? (_._1.isDefined) | (c => {
      val Some(matchStr) = c._1

      val (day, time) = (matchStr.group(1), matchStr.group(2))

      (
        new DateTime(
          year.toInt, month.toInt, day.toInt,
          time.take(2).toInt, time.slice(2, 4).toInt, time.takeRight(2).toInt),
        c._2
      )
    })).toStream
  }

  /**
    * Load paths to [[SDO]] images from the disk.
    *
    * @param sdo_files_path The base directory in which sdo data is stored.
    *
    * @param year_month The year-month from which the images were taken.
    *
    * @param sdo_sources A sequence of data sources i.e. each one a [[SDO]] instance.
    *
    * @param dirTreeCreated If SDO directory structure has been
    *                       created inside the soho_files_path,
    *                       defaults to true.
    *
    * @return Time stamped images for each soho instrument and image resolution.
    * */
  def load_images(
    sdo_files_path: Path, year_month: YearMonth,
    sdo_sources: Seq[SDO], dirTreeCreated: Boolean): Iterable[(DateTime, (SDO, Path))] = {


    val (year, month) = (year_month.getYear.toString, "%02d".format(year_month.getMonthOfYear))
    val filePattern = getFilePattern(year_month, sdo_sources)

    val image_paths = if(dirTreeCreated) {
      ls! sdo_files_path |? (s => s.isDir && sdo_sources.map(_.instrument).contains(s.segments.last)) ||
        (d => {
          ls! d |?
            (s => s.isDir && s.segments.contains(year)) ||
            (ls! _) |?
            (_.segments.contains(month))
        }) ||
        (ls! _ )
    } else {
      ls.rec! sdo_files_path
    }

    (image_paths | (file => {
      (filePattern.findFirstMatchIn(file.segments.last), file)
    }) |? (_._1.isDefined) | (c => {
      val Some(matchStr) = c._1

      val (day, time, source_str) = (matchStr.group(1), matchStr.group(2), matchStr.group(3))

      val source = SDO(source_str.split("_").head, source_str.split("_").last.toInt)

      (
        new DateTime(
          year.toInt, month.toInt, day.toInt,
          time.take(2).toInt, time.slice(2, 4).toInt, time.takeRight(2).toInt),
        (source, c._2)
      )
    })).toStream
  }


}