package io.github.mandar2812.PlasmaML.helios.data

import ammonite.ops.Path
import org.jsoup.Jsoup
import collection.JavaConverters._

object GOESData {

  val base_url = "https://satdat.ngdc.noaa.gov/sem/goes/data/avg/"

  object Quantities {

    val PROTON_FLUX_1m = "_eps_1m_"
    val PROTON_FLUX_5m = "_eps_5m_"

    val MAG_FIELD_1m = "_magneto_1m"
    val MAG_FIELD_5m = "_magneto_5m"

    val XRAY_FLUX_1m = "_xrs_1m"
    val XRAY_FLUX_5m = "_xrs_5m"
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

        elements_mission.map(_.attr("href")).filter(_.contains(quantity)).toList
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


}