package io.github.mandar2812.PlasmaML

import ammonite.ops.Path
import io.github.mandar2812.PlasmaML.helios.data.{SDOLoader, SOHOLoader}
import org.joda.time.LocalDate

package object helios {

  def download(source: data.Source, download_path: Path)(start: LocalDate, end: LocalDate): Unit = source match {

    case data.SOHO(instrument, size) =>
      SOHOLoader.bulk_download(download_path)(instrument, size)(start, end)

    case data.SDO(instrument, size) =>
      SDOLoader.bulk_download(download_path)(instrument, size)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }

}
