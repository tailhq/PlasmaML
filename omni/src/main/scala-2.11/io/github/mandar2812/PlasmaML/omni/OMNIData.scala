package io.github.mandar2812.PlasmaML.omni

import ammonite.ops.Path
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger
import org.joda.time.{DateTime, DateTimeZone}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

/**
  * <h3>OMNI Data Properties</h3>
  *
  * Contains static definitions of meta data for the OMNI data set.
  * @author mandar2812 date 12/7/16.
  * */
object OMNIData {

  val logger = Logger.getLogger(this.getClass)

  val base_url = "ftp://spdf.gsfc.nasa.gov/"

  val omni_uri = "pub/data/omni/low_res_omni/"

  /**
    * The column numbers of important quantities in
    * the OMNI data files.
    * */
  object Quantities {

    //Sunspot Number
    val sunspot_number = 39

    //Genmagnetic Indices
    val Dst = 40
    val AE = 41
    val Kp = 38

    //L1 quantities
    val V_SW = 24
    val B_Z = 16
    val P = 28
  }

  /**
    * Stores the missing value strings for
    * each column index in the hourly resolution
    * OMNI files.
    * */
  var columnFillValues = Map(
    16 -> "999.9", 21 -> "999.9",
    24 -> "9999.", 23 -> "999.9",
    40 -> "99999", 22 -> "9999999.",
    25 -> "999.9", 28 -> "99.99",
    27 -> "9.999", 39 -> "999",
    45 -> "99999.99", 46 -> "99999.99",
    47 -> "99999.99", 15 -> "999.9")

  /**
    * Contains the name of the quantity stored
    * by its column index (which starts from 0).
    * */
  var columnNames = Map(
    24 -> "Solar Wind Speed",
    16 -> "I.M.F Bz",
    40 -> "Dst",
    41 -> "AE",
    38 -> "Kp",
    39 -> "Sunspot Number",
    28 -> "Plasma Flow Pressure")

  //The column indices corresponding to the year, day of year and hour respectively
  val dateColumns = List(0, 1, 2)

  def getFilePattern(year: Int): String = "omni2_"+year+".csv"

}

/**
  * <h3>OMNI Data Pipelines</h3>
  *
  * Pipelines and workflows which make processing OMNI data easier.
  * @author mandar2812
  * */
object OMNILoader {

  import io.github.mandar2812.PlasmaML.omni.OMNIData._

  //Set time zone to UTC.
  DateTimeZone.setDefault(DateTimeZone.UTC)

  /**
    * Defines the date time format stored in OMNI files, i.e. Year - Day of Year - Hour
    * */
  val dt_format: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/D/H")

  def get_download_url(year: Int): String = base_url + omni_uri + s"omni2_$year.dat"

  def download(year_range: Range = 2001 to 2017, path: Path): Unit = {

    println("Downloading OMNI hourly data")
    println()

    year_range.foreach(y => {
      print("Year: ")
      pprint.pprintln(y)
      utils.downloadURL(get_download_url(y), (path/("omni2_"+y+".csv")).toString)
    })

  }
  /**
    * Returns a [[io.github.mandar2812.dynaml.pipes.DataPipe]] which
    * reads an OMNI file cleans it and extracts the columns specified
    * by targetColumn and exogenousInputs.
    * */
  def omniFileToStream(targetColumn: Int, exogenousInputs: Seq[Int]) =
    fileToStream >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      dateColumns++List(targetColumn)++exogenousInputs,
      columnFillValues) >
    removeMissingLines

  /**
    * Takes a stream of OMNI data records (each line is a comma separated string),
    * extracts the portions relevant for building the time stamp and propagates
    * time stamped data as a [[Tuple2]] of [[DateTime]] and a Scala sequence containing
    * the extracted quantities.
    * */
  val processWithDateTime = IterableDataPipe((line: String) => {
    val splits = line.split(",")
    val num_splits = splits.length

    val dt_string = splits.take(3).mkString("/")
    val data = splits.takeRight(num_splits - 3).map(_.toDouble)
    val timestamp = DateTime.parse(dt_string, dt_format)
    (timestamp, data.toSeq)
  })

  /**
    * Returns a pipe which constructs a running forward looking time window
    * for a stream of date time stamped univariate values.
    *
    * <b>Usage</b>: For obtaining a pipe which takes (t, y(t)) and returns (t, [y(t+p), ..., y(t+p+n)])
    * {{{
    *   val p = 5
    *   val n = 10
    *   val fpipe = OMNIData.forward_time_window(n, p)
    * }}}
    * */
  val forward_time_window = MetaPipe21(
    (deltaT: Int, timelag: Int) => (lines: Iterable[(DateTime, Double)]) =>
      lines.toList.sliding(deltaT+timelag+1).map(history => {

        val features: Seq[Double] = history.slice(deltaT, deltaT+timelag).map(_._2)

        (history.head._1, features)
      }).toStream
  )

  /**
    * Returns a pipe which constructs a running forward looking time window
    * for a stream of date time stamped univariate values.
    *
    * <b>Usage</b>: For obtaining a pipe which takes (t, y(t)) and
    * returns (t, ([y(t-h+1), ..., y(t)], [y(t+p), ..., y(t+p+n)]))
    *
    * {{{
    *   val p = 5
    *   val n = 10
    *   val fpipe = OMNIData.forward_and_backward_time_window(h, (n, p))
    * }}}
    * */
  val forward_and_backward_time_window = MetaPipe21(
    (past: Int, future: (Int, Int)) => (lines: Iterable[(DateTime, Double)]) =>
      lines.toList.sliding(past+future._1+future._2+1).map(history => {

        val features: Seq[Double] = history.slice(past+future._1, past+future._1+future._2).map(_._2)
        val features_history: Seq[Double] = history.slice(0, past+1).map(_._2)

        (history(past)._1, (features_history, features))
      }).toStream
  )

  /**
    * Constructs a running forward looking time window
    * for a stream of date time stamped multivariate values.
    * <b>Usage</b>: See [[forward_time_window]].
    * */
  val mv_forward_time_window = MetaPipe21(
    (deltaT: Int, timelag: Int) => (lines: Iterable[(DateTime, Seq[Double])]) => {
      val num_quantities = lines.head._2.length

      lines.toList.sliding(deltaT+timelag+1).map((history) => {

        val features: Seq[Seq[Double]] = history.slice(deltaT, deltaT+timelag).map(_._2)

        val proc_features: Seq[Seq[Double]] = (0 until num_quantities).map(q => {
          features.map(_(q))
        })

        (history.head._1, proc_features)
      }).toStream
    }
  )

  /**
    * Returns a [[DataPipe]] which takes a list of omni csv file paths,
    * extracts a chosen quantity as a sliding time series.
    *
    * @param deltaT The minimum time lag from which each window should start.
    * @param time_window The size of the time window.
    * @param targetColumn The quantity to extract,
    *                     specified as the column number in the omni file,
    *                     see [[OMNIData.Quantities]]
    *
    * <b>Usage</b>: A pipe which extracts a sliding time series of Dst,
    * when given a list of file paths.
    * {{{
    *   val omni_process_pipe = OMNILoader.omniVarToSlidingTS(5, 10)(OMNIData.Quantities.Dst)
    * }}}
    * */
  def omniVarToSlidingTS(deltaT: Int, time_window: Int)(targetColumn: Int = OMNIData.Quantities.V_SW) =
    IterableFlatMapPipe(omniFileToStream(targetColumn, Seq())) >
      processWithDateTime >
      IterableDataPipe((p: (DateTime, Seq[Double])) => (p._1, p._2.head)) >
      forward_time_window(deltaT, time_window)

  /**
    * Returns a [[DataPipe]] which takes a list of omni csv file paths,
    * extracts a chosen quantity as a sliding time series.
    *
    * @param deltaT The minimum time lag from which each window should start.
    * @param time_window The size of the time window.
    * @param targetColumn The quantity to extract,
    *                     specified as the column number in the omni file,
    *                     see [[OMNIData.Quantities]]
    *
    * <b>Usage</b>: A pipe which extracts a sliding time series of Dst,
    * when given a list of file paths.
    * {{{
    *   val omni_process_pipe = OMNILoader.omniVarToSlidingTS(5, 10)(OMNIData.Quantities.Dst)
    * }}}
    * */
  def omniVarToSlidingTS(
    past: Int, deltaT: Int,
    time_window: Int)(
    targetColumn: Int) =
    StreamFlatMapPipe(omniFileToStream(targetColumn, Seq())) >
      processWithDateTime >
      IterableDataPipe((p: (DateTime, Seq[Double])) => (p._1, p._2.head)) >
      forward_and_backward_time_window(past, (deltaT, time_window))

  /**
    * Returns a [[DataPipe]] which takes a list of omni csv file paths,
    * extracts a list of quantities as a sliding time series.
    *
    * @param deltaT The minimum time lag from which each window should start.
    * @param time_window The size of the time window.
    * @param targetColumn The quantity to extract,
    *                     specified as the column number in the omni file,
    *                     see [[OMNIData.Quantities]]
    * @param exogenous_inputs A list of auxillary quantities to extract.
    *
    * <b>Usage</b>: A pipe which extracts a sliding time series of Dst,
    * when given a list of file paths.
    * {{{
    *   val omni_process_pipe = OMNILoader.omniVarToSlidingTS(5, 10)(OMNIData.Quantities.Dst)
    * }}}
    * */
  def omniDataToSlidingTS(deltaT: Int, time_window: Int)(
    targetColumn: Int = OMNIData.Quantities.V_SW,
    exogenous_inputs: Seq[Int] = Seq()) =
    StreamFlatMapPipe(omniFileToStream(targetColumn, exogenous_inputs)) >
      processWithDateTime >
      mv_forward_time_window(deltaT, time_window)

}
