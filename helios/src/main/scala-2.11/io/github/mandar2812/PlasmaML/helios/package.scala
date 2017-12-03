package io.github.mandar2812.PlasmaML

import ammonite.ops.Path
import io.github.mandar2812.PlasmaML.helios.data.{GOESLoader, SDOLoader, SOHOLoader}
import org.joda.time.{LocalDate, YearMonth}

package object helios {

  /**
    * Download solar images from a specified source.
    *
    * @param source An instance of [[data.Source]], can be constructed
    *               using any of its subclasses,
    *               ex: [[data.SOHO]] or [[data.SDO]]
    *
    * @param download_path The location on disk where the data
    *                      is to be dumped.
    *
    * @param start The starting date from which images are extracted.
    *
    * @param end The date up to which images are extracted.
    * */
  def download_image_data(
    source: data.Source, download_path: Path)(
    start: LocalDate, end: LocalDate): Unit = source match {

    case data.SOHO(instrument, size) =>
      SOHOLoader.bulk_download(download_path)(instrument, size)(start, end)

    case data.SDO(instrument, size) =>
      SDOLoader.bulk_download(download_path)(instrument, size)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }

  /**
    * Download solar flux from the GOES data repository.
    *
    * @param source An instance of [[data.Source]], can be constructed
    *               using any of its subclasses,
    *               ex: [[data.GOES]]
    *
    * @param download_path The location on disk where the data
    *                      is to be dumped.
    *
    * @param start The starting year-month from which images are extracted.
    *
    * @param end The year-month up to which images are extracted.
    *
    * @throws Exception if the data source is not valid
    *                   (i.e. not [[data.GOES]])
    * */
  def download_flux_data(
    source: data.Source, download_path: Path)(
    start: YearMonth, end: YearMonth): Unit = source match {

    case data.GOES(quantity, format) =>
      GOESLoader.bulk_download(download_path)(quantity, format)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }

}
