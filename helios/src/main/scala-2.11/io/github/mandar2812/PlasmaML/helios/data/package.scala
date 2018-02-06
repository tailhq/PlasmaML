package io.github.mandar2812.PlasmaML.helios

import ammonite.ops.Path
import io.github.mandar2812.PlasmaML.omni.OMNIData
import io.github.mandar2812.dynaml.utils
import org.joda.time.{Duration, LocalDate, Period, YearMonth}

/**
  * <h3>Helios Data Facility</h3>
  *
  * Package enabling downloading and storage of heliospheric image data.
  *
  * @author mandar2812
  * */
package object data {

  /**
    * Download a resource (image, file) from a sequence of urls to a specified
    * disk location.
    * */
  def download_batch(path: Path)(urls: List[String]): Unit = {
    urls.par.foreach(s => utils.downloadURL(s, (path/s.split('/').last).toString()))
  }

  /**
    * Perform a bulk download of images within some date range
    * */
  def download_day_range(download: (LocalDate) => Unit)(start: LocalDate, end: LocalDate): Unit = {

    val num_days = new Duration(start.toDateTimeAtStartOfDay, end.toDateTimeAtStartOfDay).getStandardDays.toInt

    (0 to num_days).map(start.plusDays).par.foreach(download)
  }

  def download_month_range(download: (YearMonth) => Unit)(start: YearMonth, end: YearMonth): Unit = {

    val period = new Period(
      start.toLocalDate(1).toDateTimeAtStartOfDay,
      end.toLocalDate(31).toDateTimeAtStartOfDay)

    val num_months = (12*period.getYears) + period.getMonths

    (0 to num_months).map(start.plusMonths).par.foreach(download)
  }

  sealed trait Source
  sealed trait SolarImagesSource extends Source

  case class SOHO(instrument: String, size: Int = SOHOData.Resolutions.s512) extends SolarImagesSource
  case class SDO(instrument: String, size: Int = SDOData.Resolutions.s512) extends SolarImagesSource

  case class GOES(
    quantity: String = GOESData.Quantities.XRAY_FLUX_5m,
    format: String = GOESData.Formats.CSV) extends Source

  case class OMNI(quantity: Int = OMNIData.Quantities.V_SW) extends Source

}
