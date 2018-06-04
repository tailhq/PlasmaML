import ammonite.ops._
import org.joda.time._
import io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.data.{SOHO, SOHOData}
import io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time.format.DateTimeFormat

DateTimeZone.setDefault(DateTimeZone.UTC)

/**
  * Perform a bulk download of the SOHO archive.
  *
  * @param start_date Download starting from a date string "yyyy-mm-dd"
  * @param end_date Download until a date string "yyyy-mm-dd"
  * @param path Destination directory to put data
  * @param size Resolution of the images to be downloaded, 512 or 1024
  * @param instrument The SOHO instrument from which to download,
  *                   specified as a string. See [[SOHOData.Instruments]]
  * */
@main
def apply(start_date: String, end_date: String, path: Path, size: Int, instrument: String): Unit = {
  val formatter = DateTimeFormat.forPattern("YYYY-MM-dd")

  val start_dt = formatter.parseLocalDate(start_date)
  val end_dt   = formatter.parseLocalDate(end_date)

  helios.download_image_data(SOHO(instrument, size), path)(start_dt, end_dt)
}
