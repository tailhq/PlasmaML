package io.github.mandar2812.PlasmaML.helios.data

import ammonite.ops._
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}
import org.joda.time.{DateTime, DateTimeZone, Instant, YearMonth}
import org.jsoup.Jsoup

import collection.JavaConverters._
import scala.util.matching.Regex

/**
  * Helper object for downloading solar images from the
  * <a href="http://www.swpc.noaa.gov/products/goes-x-ray-flux">GOES</a> archive.
  * @author mandar2812 date 27/11/2017
  * */
object GOESData {

  val base_url = "https://satdat.ngdc.noaa.gov/sem/goes/data/avg/"

  object Quantities {

    val PROTON_FLUX_1m = "eps_1m_"
    val PROTON_FLUX_5m = "eps_5m_"

    val MAG_FIELD_1m = "magneto_1m"
    val MAG_FIELD_5m = "magneto_5m"

    val XRAY_FLUX_1m = "xrs_1m"
    val XRAY_FLUX_5m = "xrs_5m"
  }

  object Formats {
    val CSV = "csv"
    val NETCDF = "cdf"
  }

  //A regular expression which extracts the data segment of
  //a goes csv file.
  val cleanRegex: Regex = """time_tag,xs,xl([.\s\w$,-:]+)""".r

  val missingValue: String = "-99999.0"

  def getFilePattern(date: YearMonth, source: GOES): Regex = {
    val (year, month) = (date.getYear.toString, date.getMonthOfYear.toString)

    ("""g\w+_""" +source.quantity+"""_"""+year+month+"""\d{2}?_"""+year+month+"""\d{2}?\."""+source.format).r
  }

}

object GOESLoader {

  import GOESData._

  DateTimeZone.setDefault(DateTimeZone.UTC)

  /**
    * Download all the available images
    * for a given date, corresponding to
    * some specified instrument code.
    * */
  def fetch_urls(path: Path)(quantity: String, format: String)(year: Int, month: Int) = {

    //Construct the url to download file manifest for date in question.
    val download_url = base_url+year+"/"+"%02d".format(month)+"/"

    val hrefs = try {
      val doc = Jsoup.connect(download_url).timeout(0).get()
      //Extract the elements containing the data file urls
      val elements = doc.select("a[href]")
        .iterator()
        .asScala

      elements.map(_.attr("href")).filter(_.split('/').head.contains("goes")).flatMap(link => {

        val doc_mission = Jsoup.connect(download_url+link+format+"/").timeout(0).get()
        //Extract the elements containing the data file urls
        val elements_mission = doc_mission.select("a[href]")
          .iterator()
          .asScala

        elements_mission.map(h => link+format+"/"+h.attr("href")).filter(_.contains(quantity)).toList
      }).toList

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
    * Download GOES data from a specified month YYYY-MM.
    *
    * @param path The root path where the data will be downloaded,
    *             the downloader appends soho/[instrument]/[year] to
    *             the path supplied and places the images in there if
    *             the createDirTree flag is set to true.
    *
    * @param createDirTree If this is set to false, then the images
    *                      are placed directly in the path supplied.
    *
    * @param quantity The quantity to download, see [[GOESData.Quantities]]
    *
    * @param format The format of the data, can be either "csv" or "netcdf",
    *               see [[GOESData.Formats]]
    *
    * @param date a Joda time [[YearMonth]] instance.
    * */
  def download(
    path: Path, createDirTree: Boolean = true)(
    quantity: String, format: String = GOESData.Formats.CSV)(
    date: YearMonth): Unit = {

    val (year, month) = (date.getYear, date.getMonthOfYear)

    val download_path = if(createDirTree) path/'goes/quantity/year.toString/"%02d".format(month) else path

    if(!(exists! download_path)) {
      mkdir! download_path
    }

    println("Downloading GOES data from the NOAA Archive for: "+date+"\nDownload Path: "+download_path)

    download_batch(download_path)(fetch_urls(path)(quantity, format)(year, month))
  }

  /**
    * Perform a bulk download of GOES data within some year-month range
    * */
  def bulk_download(
    path: Path, createDirTree: Boolean = true)(
    instrument: String, format: String = GOESData.Formats.CSV)(
    start: YearMonth, end: YearMonth): Unit = {

    download_month_range(download(path, createDirTree)(instrument, format))(start, end)
  }

  /**
    * Parse a GOES X-Ray flux csv file.
    *
    * @param file The path of the data file as an
    *             ammonite [[Path]].
    *
    * @return A sequence of time stamped x-ray fluxes.
    * */
  def parse_file(file: Path): Stream[(DateTime, (Double, Double))] = {

    /*
    * The parsing follows in three steps:
    *
    *   1. Read the file into a string
    *
    *   2. Extract the data section of the csv,
    *      discarding the meta-data,
    *      using a regular expression match.
    *
    *   3. Extract the x-ray fluxes in a collection
    *      along with the time stamps.
    *
    * */

    (read! file |>
      ((c) => cleanRegex.findAllIn(c).matchData.map(_.group(1)).toList.head.split("\\r?\\n").drop(1)) |
      (line => {
        val splits = line.split(',')

        val date_time = DateTime.parse(splits.head, DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss.SSS"))

        val xray_low_freq = if (splits(1) == missingValue) Double.NaN else splits(1).toDouble

        val xray_high_freq = if (splits.last == missingValue) Double.NaN else splits.last.toDouble

        (date_time, (xray_low_freq, xray_high_freq))
      })).toStream
  }

  def load_goes_data(
    goes_files_path: Path, year_month: YearMonth,
    goes_source: GOES, dirTreeCreated: Boolean = true): Stream[(DateTime, Seq[(Double, Double)])] = {

    val (year, month) = (year_month.getYear.toString, "%02d".format(year_month.getMonthOfYear))

    val filePattern = getFilePattern(year_month, goes_source)

    val files = if(dirTreeCreated) {

      ls! goes_files_path/goes_source.quantity/year/month

    } else {
      ls! goes_files_path
    }

    files |?
      (_.isFile) |?
      (f => filePattern.findFirstMatchIn(f.segments.last).isDefined) ||
      parse_file |>
      (_.groupBy(_._1).mapValues(_.map(_._2)).toStream.sortBy(_._1.getMillis))
  }

}