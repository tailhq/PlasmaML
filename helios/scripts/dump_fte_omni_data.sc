import _root_.ammonite.ops._
import _root_.org.joda.time._
import _root_.org.joda.time.format.DateTimeFormat
import _root_.breeze.linalg._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.data._
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}

DateTimeZone.setDefault(DateTimeZone.UTC)

def solar_wind_time_series(
  start: DateTime,
  end: DateTime
): ZipDataSet[DateTime, Seq[Double]] = {

  val omni_data_path = pwd / 'data

  val load_omni_file =
    fileToStream >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        OMNIData.dateColumns ++ List(OMNIData.Quantities.V_SW),
        OMNIData.columnFillValues
      ) >
      OMNILoader.processWithDateTime

  dtfdata
    .dataset(start.getYear to end.getYear)
    .map(
      DataPipe(
        (i: Int) => omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
      )
    )
    .flatMap(load_omni_file)
    .to_zip(identityPipe[(DateTime, Seq[Double])])

}

val deltaTFTE      = 0
val fteStep        = 0
val fte_data_path  = home / 'Downloads / 'fte
val log_scale_fte  = false
val latitude_limit = 90d
val conv_flag      = false
val log_scale_omni = false

val start = new DateTime(2007, 1, 1, 0, 0, 0)
val end   = new DateTime(2018, 12, 31, 23, 59, 59)

val fte_data = fte.data.load_fte_data_bdv(
  fte_data_path,
  fte.data.carrington_rotations,
  log_scale_fte,
  start,
  end
)(deltaTFTE, fteStep, latitude_limit, conv_flag)

val omni_data =
  solar_wind_time_series(start, end)

val data = fte_data.join(omni_data)

type PAT = (DateTime, (DenseVector[Double], Seq[Double]))

val dump_file = home / 'Downloads / s"fte_sw_${start.getYear}-${end.getYear}.csv"

data.foreach(DataPipe((p: PAT) => {
  
  val line = Seq(p._1.toString("YYYY-MM-dd-HH"), p._2._1.toArray.mkString(","), p._2._2.mkString(",")).mkString(",")+"\n"

  write.append(
    dump_file,
    line
  )
}))
