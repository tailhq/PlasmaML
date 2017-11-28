package io.github.mandar2812.PlasmaML.helios.data

import ammonite.ops._
import org.joda.time.YearMonth
import org.jsoup.Jsoup

import collection.JavaConverters._

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
    val NETCDF = "netcdf"
  }
}

object GOESLoader {
  import GOESData._


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


}