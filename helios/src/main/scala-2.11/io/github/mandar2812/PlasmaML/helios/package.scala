package io.github.mandar2812.PlasmaML

import ammonite.ops.Path
import io.github.mandar2812.PlasmaML.helios.data.{GOESLoader, SDOLoader, SOHOLoader}
import org.joda.time.{LocalDate, YearMonth}

package object helios {

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

  def download_flux_data(
    source: data.Source, download_path: Path)(
    start: YearMonth, end: YearMonth): Unit = source match {

    case data.GOES(quantity, format) =>
      GOESLoader.bulk_download(download_path)(quantity, format)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }
}
