package io.github.mandar2812.PlasmaML.helios.fte

import ammonite.ops._
import org.joda.time._
import org.joda.time.format.DateTimeFormat
import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.math._
import breeze.numerics._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.tensorflow.data._
import io.github.mandar2812.dynaml.tensorflow.utils._
import io.github.mandar2812.dynaml.tensorflow.{dtf, dtfdata, dtflearn, dtfutils}
import io.github.mandar2812.dynaml.tensorflow.implicits._
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.{utils => dutils}
import io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import org.platanios.tensorflow.api._
import _root_.org.json4s._
import _root_.org.json4s.JsonDSL._
import _root_.org.json4s.jackson.Serialization.{
  read => read_json,
  write => write_json
}
import org.json4s.jackson.JsonMethods._
import nom.tam.fits._

package object data {

  //Set time zone to UTC
  DateTimeZone.setDefault(DateTimeZone.UTC)

  implicit val formats = DefaultFormats + FieldSerializer[Map[String, Any]]()

  case class FTEConfig(
    data_limits: (Int, Int),
    deltaTFTE: Int,
    fteStep: Int,
    latitude_limit: Double,
    log_scale_fte: Boolean)

  case class OMNIConfig(
    deltaT: (Int, Int),
    log_flag: Boolean,
    quantity: Int = OMNIData.Quantities.V_SW,
    use_persistence: Boolean = false)

  case class FteOmniConfig(
    fte_config: FTEConfig,
    omni_config: OMNIConfig,
    multi_output: Boolean = true,
    probabilistic_time_lags: Boolean = true,
    timelag_prediction: String = "mode",
    fraction_variance: Double = 1d)
      extends helios.Config

  def read_exp_config(file: Path): Option[FteOmniConfig] =
    if (exists ! file) {
      try {
        val config = parse((read.lines ! file).head).values
          .asInstanceOf[Map[String, Any]]
        val fte_config  = config("fte_config").asInstanceOf[Map[String, Any]]
        val omni_config = config("omni_config").asInstanceOf[Map[String, Any]]

        val omni_deltaT = omni_config("deltaT")
          .asInstanceOf[Map[String, BigInt]]
        val fte_data_limits = fte_config("data_limits")
          .asInstanceOf[Map[String, BigInt]]

        Some(
          FteOmniConfig(
            FTEConfig(
              data_limits = (
                fte_data_limits("_1$mcI$sp").toInt,
                fte_data_limits("_2$mcI$sp").toInt
              ),
              fte_config("deltaTFTE").asInstanceOf[BigInt].toInt,
              fte_config("fteStep").asInstanceOf[BigInt].toInt,
              fte_config("latitude_limit").asInstanceOf[Double],
              fte_config("log_scale_fte").asInstanceOf[Boolean]
            ),
            OMNIConfig(
              (
                omni_deltaT("_1$mcI$sp").toInt,
                omni_deltaT("_2$mcI$sp").toInt
              ),
              omni_config("log_flag").asInstanceOf[Boolean],
              omni_config("quantity").asInstanceOf[BigInt].toInt,
              omni_config("use_persistence").asInstanceOf[Boolean]
            ),
            config("multi_output").asInstanceOf[Boolean],
            config("probabilistic_time_lags").asInstanceOf[Boolean],
            config("timelag_prediction").asInstanceOf[String],
            config("fraction_variance").asInstanceOf[Double]
          )
        )

      } catch {
        case _: Exception => None
      }
    } else {
      None
    }

  def write_exp_config(config: FteOmniConfig, dir: Path): Unit = {
    if (!(exists ! dir / "config.json")) {
      val config_json = write_json(config)
      write(dir / "config.json", config_json)
    }
  }

  //Load the Carrington Rotation Table
  val carrington_rotation_table: Path = pwd / 'data / "CR_Table.rdb"

  case class CarringtonRotation(start: DateTime, end: DateTime) {

    def contains(dt: DateTime): Boolean = dt.isAfter(start) && dt.isBefore(end)
  }

  val carrington_rotations: ZipDataSet[Int, CarringtonRotation] =
    dtfdata
      .dataset(pipes.process_carrington_file(carrington_rotation_table))
      .to_zip(pipes.read_time_stamps)

//Some hard-coded meta data for FTE/Bss files.
//The latitude discretization
  val latitude_grid = {
    dutils
      .range[Double](-1d, 1d, 360)
      .map(math.asin)
      .map(math.toDegrees)
      .zipWithIndex
      .filter(c => c._2 > 0 && c._2 <= 359 && c._2 % 2 == 1)
      .map(_._1)
      .map(x => BigDecimal.binary(x, new java.math.MathContext(4)).toDouble)
  }

//The longitude discretization
  val longitude_grid = (1 to 360).map(_.toDouble)

  case class HelioPattern(data: (Double, Double, Option[Double]))
      extends AnyVal {

    def _1: Double         = data._1
    def _2: Double         = data._2
    def _3: Option[Double] = data._3

    def lat: Double           = _2
    def lon: Double           = _1
    def value: Option[Double] = _3
  }

  object pipes {

    val read_time_stamps = DataPipe((s: Array[String]) => {

      val datetime_pattern = "YYYY.MM.dd_HH:mm:ss"
      val dt               = format.DateTimeFormat.forPattern(datetime_pattern)

      val limits = (DateTime.parse(s(1), dt), DateTime.parse(s(3), dt))

      (s.head.toInt, CarringtonRotation(limits._1, limits._2))
    })

    val read_lines_gong = (gong_file: Path) =>
      (read.lines ! gong_file).toIterable.drop(3)

    val read_lines_hmi = (hmi_file: Path) =>
      (read.lines ! hmi_file).toIterable.drop(4)

    val process_carrington_file: DataPipe[Path, Iterable[Array[String]]] =
      DataPipe((p: Path) => (read.lines ! p).toStream) >
        dropHead >
        dropHead >
        trimLines >
        replaceWhiteSpaces >
        splitLine

    val fte_file = MetaPipe(
      (data_path: Path) =>
        (carrington_rotation: Int) => {
          val hmi_file  = data_path / s"HMIfootpoint_ch_csss${carrington_rotation}HR.dat"
          val gong_file = data_path / s"GONGfootpoint_ch_csss${carrington_rotation}HR.txt"

          if (exists ! gong_file) read_lines_gong(gong_file)
          else read_lines_hmi(hmi_file)
        }
    )

    val brss_file = MetaPipe(
      (data_path: Path) =>
        (carrington_rotation: Int) =>
          data_path / s"GONGbrss_csss${carrington_rotation}HR.fits"
    )

    val fits_file_to_array = DataPipe((file: Path) => {
      new Fits(file.toString)
        .getHDU(0)
        .getKernel()
        .asInstanceOf[Array[Array[Float]]]
        .toIterable
    })

    val fits_array_to_collection = DataPipe(
      (arr: Iterable[Array[Float]]) =>
        latitude_grid
          .zip(arr)
          .flatMap(
            (s: (Double, Array[Float])) =>
              longitude_grid
                .zip(s._2)
                .map(
                  (p: (Double, Float)) =>
                    HelioPattern(p._1, s._1, Some(p._2.toDouble))
                )
          )
          .toIterable
    )

    val read_brss_file = brss_file >> (fits_file_to_array > fits_array_to_collection)

    val read_fte_file = {
      fte_file >> (
        trimLines >
          replaceWhiteSpaces >
          splitLine >
          IterableDataPipe((s: Array[String]) => s.length == 5) >
          IterableDataPipe((s: Array[String]) => {
            val (lon, lat) = (s.head.toDouble, s(1).toDouble)
            val fte: Option[Double] = try {
              Some(s(2).toDouble)
            } catch {
              case _: Exception => None
            }

            HelioPattern(lon, lat, fte)

          })
      )

    }

    val clamp_fte: DataPipe[HelioPattern, HelioPattern] = DataPipe(
      (p: HelioPattern) =>
        p._3 match {
          case Some(f) =>
            if (math.abs(f) <= 1000d) p
            else HelioPattern(p.lon, p.lat, Some(1000d * math.signum(f)))
          case None => p
        }
    )

    val log_transformation = (log_flag: Boolean) =>
      (x: Double) =>
        if (log_flag) {
          if (math.abs(x) < 1d) 0d
          else math.log10(math.abs(x))
        } else x

    def log_fte(
      log_flag: Boolean
    ): DataPipe[HelioPattern, HelioPattern] = DataPipe(
      p =>
        HelioPattern((p.lon, p.lat, p.value.map(log_transformation(log_flag))))
    )

    val process_timestamps_rotation: DataPipe[
      (Int, (CarringtonRotation, Iterable[HelioPattern])),
      Iterable[(DateTime, HelioPattern)]
    ] = DataPipe(
      (rotation_data) => {

        val (_, (rotation, data)) = rotation_data

        val duration = new Duration(rotation.start, rotation.end)

        val time_jump = duration.getMillis / 360.0

        val time_stamp = (p: HelioPattern) =>
          rotation.end.toInstant
            .minus((time_jump * p.lon).toLong)
            .toDateTime

        data.map(
          p =>
            (
              time_stamp(p),
              p
            )
        )

      }
    )

    val process_timestamps_rotation2: DataPipe[
      (
        Int,
        (CarringtonRotation, (Iterable[HelioPattern], Iterable[HelioPattern]))
      ),
      (Iterable[(DateTime, HelioPattern)], Iterable[(DateTime, HelioPattern)])
    ] = DataPipe(
      (rotation_data) => {

        val (_, (rotation, (fte, brss))) = rotation_data

        val duration = new Duration(rotation.start, rotation.end)

        val time_jump = duration.getMillis / 360.0

        val time_stamp = (p: HelioPattern) =>
          rotation.end.toInstant
            .minus((time_jump * p.lon).toLong)
            .toDateTime

        (
          fte.map(
            p =>
              (
                time_stamp(p),
                p
              )
          ),
          brss.map(
            p =>
              (
                time_stamp(p),
                p
              )
          )
        )

      }
    )

    val round_datetime_to_hour: DataPipe[DateTime, DateTime] = DataPipe(
      (d: DateTime) =>
        new DateTime(
          d.getYear,
          d.getMonthOfYear,
          d.getDayOfMonth,
          d.getHourOfDay,
          0,
          0
        )
    )

    def crop_data_by_latitude(latitude_limit: Double) = DataPipe(
      (pattern: (DateTime, Seq[HelioPattern])) =>
        (
          pattern._1,
          pattern._2.filter(ftep => math.abs(ftep.lat) <= latitude_limit)
        )
    )

    val load_slice_to_bdv =
      DataPipe[(DateTime, Seq[HelioPattern]), (DateTime, DenseVector[Double])](
        (s: (DateTime, Seq[HelioPattern])) => {

          val num_days_year =
            new DateTime(s._1.getYear, 12, 31, 23, 59, 0).getDayOfYear()
          val t: Double = s._1.getDayOfYear.toDouble / num_days_year
          val xs: Seq[Double] = Seq(t, s._2.head._1) ++ s._2
            .map(_._3.get)
          (s._1, DenseVector(xs.toArray))
        }
      )

    val sort_by_date =
      DataPipe[
        Iterable[(DateTime, Seq[HelioPattern])],
        Iterable[(DateTime, Seq[HelioPattern])]
      ](
        _.toSeq.sortBy(_._1)
      )

    val group_by_time_sort_by_latitude =
      DataPipe[
        Iterable[(DateTime, HelioPattern)],
        Iterable[(DateTime, Seq[HelioPattern])]
      ](
        _.groupBy(_._1).map(p => (p._1, p._2.map(_._2).toSeq.sortBy(_.lat)))
      )

    val sort_data =
      IterableDataPipe(round_datetime_to_hour * identityPipe[HelioPattern]) >
        group_by_time_sort_by_latitude >
        sort_by_date

    val zip_fte_brss = DataPipe2(
      (
        fte: Iterable[(DateTime, Seq[HelioPattern])],
        brss: Iterable[(DateTime, Seq[HelioPattern])]
      ) => fte.zip(brss).map(p => (p._1._1, p._1._2 ++ p._2._2))
    )

  }

  /**
    * Load the Flux Tube Expansion (FTE) data.
    *
    * Assumes the files have the schema
    * "HMIfootpoint_ch_csss{carrington_rotation}HR.dat"
    *
    * @param data_path Path containing the FTE data files.
    * @param cr The Carrington rotation number
    *
    * @throws java.nio.file.NoSuchFileException if the files cannot be
    *                                           found in the specified location
    * */
  def get_fte_for_rotation(
    data_path: Path
  )(cr: Int
  ): Iterable[(Int, Iterable[HelioPattern])] =
    try {
      Iterable((cr, pipes.read_fte_file(data_path)(cr)))
    } catch {
      case _: java.nio.file.NoSuchFileException => Iterable()
    }

  /**
    * Load the source surface radial magnetic field (Brss) data.
    *
    * Assumes the files have the schema
    * ""GONGbrss_csss{carrington_rotation}HR.fits""
    *
    * @param data_path Path containing the FTE data files.
    * @param cr The Carrington rotation number
    *
    * @throws java.nio.file.NoSuchFileException if the files cannot be
    *                                           found in the specified location
    * */
  def get_brss_for_rotation(
    data_path: Path
  )(cr: Int
  ): Iterable[(Int, Iterable[HelioPattern])] =
    try {
      Iterable((cr, pipes.read_brss_file(data_path)(cr)))
    } catch {
      case _: java.nio.file.NoSuchFileException => Iterable()
    }

  implicit val dateOrdering: Ordering[DateTime] = new Ordering[DateTime] {
    override def compare(x: DateTime, y: DateTime): Int =
      if (x.isBefore(y)) -1 else 1
  }

  /**
    * Creates a DynaML data set consisting of time FTE values.
    * The FTE values are loaded in a [[Tensor]] object.
    * */
  def load_fte_data(
    data_path: Path,
    carrington_rotation_table: ZipDataSet[Int, CarringtonRotation],
    log_flag: Boolean,
    start: DateTime,
    end: DateTime
  )(deltaTFTE: Int,
    fte_step: Int,
    latitude_limit: Double,
    conv_flag: Boolean
  ): ZipDataSet[DateTime, Tensor[Double]] = {

    val start_rotation =
      carrington_rotation_table.filter(_._2.contains(start)).data.head._1

    val end_rotation =
      carrington_rotation_table.filter(_._2.contains(end)).data.head._1

    val load_fte_for_rotation = DataPipe(get_fte_for_rotation(data_path) _)

    val fte = dtfdata
      .dataset(start_rotation to end_rotation)
      .flatMap(load_fte_for_rotation)
      .map(
        identityPipe[Int] * IterableDataPipe[HelioPattern, HelioPattern](
          pipes.clamp_fte
        )
      )
      .to_zip(identityPipe[(Int, Iterable[HelioPattern])])

    val fte_data = carrington_rotation_table.join(fte)

    val load_slice_to_tensor = DataPipe[Seq[HelioPattern], Tensor[Double]](
      (s: Seq[HelioPattern]) =>
        dtf.tensor_f64(s.length)(
          s.map(_._3.get).map(pipes.log_transformation(log_flag)): _*
        )
    )

    val processed_fte_data = {
      fte_data
        .flatMap(
          pipes.process_timestamps_rotation >
            IterableDataPipe(
              pipes.round_datetime_to_hour * identityPipe[HelioPattern]
            ) >
            pipes.group_by_time_sort_by_latitude >
            pipes.sort_by_date
        )
        .filter(DataPipe(_._2.length == 180))
        .map(pipes.crop_data_by_latitude(latitude_limit))
        .map(identityPipe[DateTime] * load_slice_to_tensor)
        .to_zip(identityPipe[(DateTime, Tensor[Double])])
    }

    println("Interpolating FTE values to fill hourly cadence requirement")

    val interpolated_fte = dtfdata.dataset(
      processed_fte_data.data
        .sliding(2)
        .filter(p => new Duration(p.head._1, p.last._1).getStandardHours > 1)
        .flatMap(i => {
          val duration  = new Duration(i.head._1, i.last._1).getStandardHours
          val delta_fte = (i.last._2 - i.head._2) / duration.toDouble

          (1 until duration.toInt)
            .map(l => (i.head._1.plusHours(l), i.head._2 + delta_fte * l))
        })
        .toIterable
    )

    val load_history = (history: Iterable[(DateTime, Tensor[Double])]) => {

      val history_size = history.toSeq.length / fte_step

      val hs = history
        .map(_._2)
        .toSeq
        .zipWithIndex
        .filter(_._2 % fte_step == 0)
        .map(_._1)

      (
        history.last._1,
        if (conv_flag)
          tfi
            .stack(hs, axis = -1)
            .reshape(history.head._2.shape ++ Shape(history_size, 1))
        else
          tfi.concatenate(hs, axis = -1)
      )
    }

    val generate_history = DataPipe(
      (s: Iterable[(DateTime, Tensor[Double])]) =>
        if (deltaTFTE > 0)
          s.sliding((deltaTFTE * fte_step) + 1).map(load_history).toIterable
        else if (conv_flag)
          s.map(c => (c._1, c._2.reshape(Shape(c._2.shape(0), 1, 1))))
        else s
    )

    processed_fte_data
      .concatenate(interpolated_fte)
      .transform(
        DataPipe[Iterable[(DateTime, Tensor[Double])], Iterable[
          (DateTime, Tensor[Double])
        ]](_.toSeq.sortBy(_._1).toIterable)
      )
      .transform(generate_history)
      .to_zip(identityPipe[(DateTime, Tensor[Double])])

  }

  /**
    * Load the OMNI solar wind time series as a [[Tensor]]
    *
    * @param start Starting time of the data.
    * @param end End time of the data.
    * @param deltaT The time window (t + l, t + l + h)
    * @param log_flag If set to true, log scale the velocity values.
    * @param quantity An integer column index corresponding to the OMNI
    *                 quantity to extract. Defaults to [[OMNIData.Quantities.V_SW]]
    *
    * @return A [[ZipDataSet]] with time indexed tensors containing
    *         sliding time histories of the solar wind.
    * */
  def load_solar_wind_data(
    start: DateTime,
    end: DateTime
  )(deltaT: (Int, Int),
    log_flag: Boolean,
    quantity: Int = OMNIData.Quantities.V_SW,
    ts_transform: DataPipe[Seq[Double], Seq[Double]] = identityPipe[Seq[Double]]
  ): ZipDataSet[DateTime, Tensor[Double]] = {

    val transform: DataPipe[Seq[Double], Seq[Double]] = if (log_flag) {
      ts_transform > DataPipe((xs: Seq[Double]) => xs.map(math.log))
    } else {
      ts_transform
    }

    val omni_processing =
      OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(quantity) >
        IterableDataPipe(
          (p: (DateTime, Seq[Double])) =>
            p._1.isAfter(start) && p._1.isBefore(end)
        ) >
        IterableDataPipe(identityPipe[DateTime] * transform)

    val omni_data_path = pwd / 'data

    dtfdata
      .dataset(start.getYear to end.getYear)
      .map(
        DataPipe(
          (i: Int) =>
            omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
        )
      )
      .transform(omni_processing)
      .to_zip(
        identityPipe[DateTime] * DataPipe(
          (s: Seq[Double]) =>
            if (s.length == 1) Tensor(s).reshape(Shape())
            else Tensor(s).reshape(Shape(s.length))
        )
      )

  }

  type SC_DATA = (
    helios.data.TF_DATA[Double, Double],
    (GaussianScalerTF[Double], GaussianScalerTF[Double])
  )

  type SC_DATA_T = (
    helios.data.TF_DATA_T[Double, Double],
    (Scaler[Tensor[Double]], GaussianScalerTF[Double])
  )

  def scale_timed_data[T: TF: IsFloatOrDouble] =
    DataPipe((dataset: helios.data.TF_DATA_T[T, T]) => {

      type P = (DateTime, (Tensor[T], Tensor[T]))

      val concat_features = tfi.stack(
        dataset.training_dataset
          .map(
            tup2_2[DateTime, (Tensor[T], Tensor[T])] > tup2_1[Tensor[T], Tensor[
              T
            ]]
          )
          .data
          .toSeq
      )

      val concat_targets = tfi.stack(
        dataset.training_dataset
          .map(
            tup2_2[DateTime, (Tensor[T], Tensor[T])] > tup2_2[Tensor[T], Tensor[
              T
            ]]
          )
          .data
          .toSeq
      )

      val n = concat_features.shape(0)

      val mean_t = concat_targets.mean(axes = 0)

      val std_t = concat_targets
        .subtract(mean_t)
        .square
        .mean(axes = 0)
        .multiply(Tensor(n / (n - 1)).castTo[T])
        .sqrt

      val mean_f = concat_features.mean(axes = 0)

      val std_f = concat_features
        .subtract(mean_f)
        .square
        .mean(axes = 0)
        .multiply(Tensor(n / (n - 1)).castTo[T])
        .sqrt

      val targets_scaler = GaussianScalerTF(mean_t, std_t)

      val features_scaler = GaussianScalerTF(mean_f, std_f)

      val scale_training_data = identityPipe[DateTime] * (features_scaler * targets_scaler)
      val scale_test_data = identityPipe[DateTime] * (features_scaler * identityPipe[
        Tensor[T]
      ])

      (
        dataset.copy(
          training_dataset = dataset.training_dataset.map(scale_training_data),
          test_dataset = dataset.test_dataset.map(scale_test_data)
        ),
        (features_scaler, targets_scaler)
      )

    })

  def scale_timed_data2[T: TF: IsFloatOrDouble](fraction: Double) =
    DataPipe((dataset: helios.data.TF_DATA_T2[DenseVector[Double], T]) => {

      type P = (DateTime, (DenseVector[Double], Tensor[T]))

      val features = dataset.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], Tensor[T])] > tup2_1[
            DenseVector[Double],
            Tensor[T]
          ]
        )
        .data

      val perform_lossy_pca = if (fraction < 1d) {
        calculatePCAScalesFeatures(false) >
          tup2_2[Iterable[DenseVector[Double]], PCAScaler] >
          compressPCA(fraction)
      } else {
        DataPipe[Iterable[DenseVector[Double]], Scaler[DenseVector[Double]]](
          _ => Scaler(identity[DenseVector[Double]])
        )
      }

      val scale_features =
        DataPipe[
          Iterable[DenseVector[Double]],
          (Iterable[DenseVector[Double]], GaussianScaler)
        ](ds => {
          val (mean, variance) = dutils.getStats(ds)
          val gs               = GaussianScaler(mean, sqrt(variance))

          (ds.map(gs(_)), gs)
        }) >
          (perform_lossy_pca * identityPipe[GaussianScaler]) >
          DataPipe2[Scaler[DenseVector[Double]], GaussianScaler, Scaler[
            DenseVector[Double]
          ]](
            (pca, gs) => gs > pca
          )

      val concat_targets = tfi.stack(
        dataset.training_dataset
          .map(
            tup2_2[DateTime, (DenseVector[Double], Tensor[T])] > tup2_2[
              DenseVector[Double],
              Tensor[
                T
              ]
            ]
          )
          .data
          .toSeq
      )

      val n = concat_targets.shape(0)

      val mean_t = concat_targets.mean(axes = 0)

      val std_t = concat_targets
        .subtract(mean_t)
        .square
        .mean(axes = 0)
        .multiply(Tensor(n / (n - 1)).castTo[T])
        .sqrt

      val targets_scaler = GaussianScalerTF(mean_t, std_t)

      val features_scaler = scale_features(features)

      val scale_training_data = identityPipe[DateTime] * (features_scaler * targets_scaler)
      val scale_test_data = identityPipe[DateTime] * (features_scaler * identityPipe[
        Tensor[T]
      ])

      (
        dataset.copy(
          training_dataset = dataset.training_dataset.map(scale_training_data),
          test_dataset = dataset.test_dataset.map(scale_test_data)
        ),
        (features_scaler, targets_scaler)
      )

    })

  def scale_dataset[T: TF: IsFloatOrDouble] =
    DataPipe((dataset: helios.data.TF_DATA[T, T]) => {

      val concat_features = tfi.stack(
        dataset.training_dataset
          .map(tup2_1[Tensor[T], Tensor[T]])
          .data
          .toSeq
      )

      val concat_targets = tfi.stack(
        dataset.training_dataset
          .map(tup2_2[Tensor[T], Tensor[T]])
          .data
          .toSeq
      )

      val n = concat_features.shape(0)

      val mean_t = concat_targets.mean(axes = 0)
      val std_t = concat_targets
        .subtract(mean_t)
        .square
        .mean(axes = 0)
        .multiply(Tensor(n / (n - 1)).castTo[T])
        .sqrt
      val mean_f = concat_features.mean(axes = 0)
      val std_f = concat_features
        .subtract(mean_f)
        .square
        .mean(axes = 0)
        .multiply(Tensor(n / (n - 1)).castTo[T])
        .sqrt

      val targets_scaler = GaussianScalerTF(mean_t, std_t)

      val features_scaler = GaussianScalerTF(mean_f, std_f)

      (
        dataset.copy(
          training_dataset =
            dataset.training_dataset.map(features_scaler * targets_scaler),
          test_dataset =
            dataset.test_dataset.map(features_scaler * identityPipe[Tensor[T]])
        ),
        (features_scaler, targets_scaler)
      )

    })

  type SCALES = (GaussianScaler, GaussianScaler)

  type MinMaxSCALES = (MinMaxScaler, MinMaxScaler)

  def scale_data(
    data: helios.data.DATA[DenseVector[Double], DenseVector[Double]]
  ): SCALES = {
    val (mean_f, sigma_sq_f) = dutils.getStats(
      data.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_1[
            DenseVector[Double],
            DenseVector[Double]
          ]
        )
        .data
    )

    val sigma_f = sqrt(sigma_sq_f)

    val (mean_t, sigma_sq_t) = dutils.getStats(
      data.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_2[
            DenseVector[Double],
            DenseVector[Double]
          ]
        )
        .data
    )

    val sigma_t = sqrt(sigma_sq_t)

    val std_training =
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Unit](
        p => {

          //Standardize features
          p._2._1 :-= mean_f
          p._2._1 :/= sigma_f

          //Standardize targets
          p._2._2 :-= mean_t
          p._2._2 :/= sigma_t

        }
      )

    val std_test =
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Unit](
        p => {

          //Standardize only features
          p._2._1 :-= mean_f
          p._2._1 :/= sigma_f

        }
      )

    data.training_dataset.foreach(std_training)
    data.test_dataset.foreach(std_test)
    (GaussianScaler(mean_f, sigma_f), GaussianScaler(mean_t, sigma_t))
  }

  def scale_data_min_max(
    data: helios.data.DATA[DenseVector[Double], DenseVector[Double]]
  ): MinMaxSCALES = {
    val (min_f, max_f) = dutils.getMinMax(
      data.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_1[
            DenseVector[Double],
            DenseVector[Double]
          ]
        )
        .data
    )

    val (min_t, max_t) = dutils.getMinMax(
      data.training_dataset
        .map(
          tup2_2[DateTime, (DenseVector[Double], DenseVector[Double])] > tup2_2[
            DenseVector[Double],
            DenseVector[Double]
          ]
        )
        .data
    )

    val (delta_f, delta_t) = (
      max_f - min_f,
      max_t - min_t
    )

    val std_training =
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Unit](
        p => {

          //Standardize features
          p._2._1 :-= min_f
          p._2._1 :/= delta_f

          //Standardize targets
          p._2._2 :-= min_t
          p._2._2 :/= delta_t

        }
      )

    val std_test =
      DataPipe[(DateTime, (DenseVector[Double], DenseVector[Double])), Unit](
        p => {

          //Standardize only features
          p._2._1 :-= min_f
          p._2._1 :/= delta_f

        }
      )

    data.training_dataset.foreach(std_training)
    data.test_dataset.foreach(std_test)
    (MinMaxScaler(min_f, max_f), MinMaxScaler(min_t, max_t))
  }

  /**
    * Creates a DynaML data set consisting of time FTE values.
    * The FTE values are loaded in a Breeze DenseVector.
    *
    * @param data_path The location containing the FTE/Brss data files
    * @param carrington_rotation_table A data collection containing meta-data
    *                                  about each Carrington Rotation
    * @param log_flag Set to true if the FTE values should be expressed in log scale
    * @param start Starting time stamp of the data
    * @param end End time stamp of the data.
    * @param deltaTFTE The size of the time history of the input features to
    *                  use when constructing input vectors.
    * @param fte_step The number of time steps to skip when constructing
    *                 each element of the time history.
    * @return A data collection consisting of (time stamp, input vector) tuples.
    */
  def load_fte_data_bdv(
    data_path: Path,
    carrington_rotation_table: ZipDataSet[Int, CarringtonRotation],
    log_flag: Boolean,
    start: DateTime,
    end: DateTime
  )(deltaTFTE: Int,
    fte_step: Int,
    latitude_limit: Double,
    conv_flag: Boolean
  ): ZipDataSet[DateTime, DenseVector[Double]] = {

    //Find which carrington rotations contain
    //the beginning and end of the data.
    val start_rotation =
      carrington_rotation_table.filter(_._2.contains(start)).data.head._1

    val end_rotation =
      carrington_rotation_table.filter(_._2.contains(end)).data.head._1

    /**
      * Data Processing: Outline.
      *
      * We start by loading the raw FTE and BRSS data for each
      * Carrington rotation. The FTE values are clamped to ensure
      * they are not greater than 1000 (physically meaningless).
      *
      */
    val data = dtfdata
      .dataset(start_rotation to end_rotation)
      .map(
        BifurcationPipe(
          identityPipe[Int],
          BifurcationPipe(
            pipes.read_fte_file(data_path),
            pipes.read_brss_file(data_path)
          )
        )
      )
      .map(
        identityPipe[Int] * (
          IterableDataPipe[HelioPattern, HelioPattern](
            pipes.clamp_fte > pipes.log_fte(log_flag)
          ) * identityPipe[Iterable[HelioPattern]]
        )
      )
      .to_zip(
        identityPipe[(Int, (Iterable[HelioPattern], Iterable[HelioPattern]))]
      )

    /**
      * After constructing `data`, the starting data collection,
      * this is processed along with the Carrington Rotation table
      * to give each data pattern a time stamp.
      *
      * t = DateTime(End of Rotation) - num_of_hours_in_rotation*Longitude/360
      *
      * After time stamping (rounding to hour), the FTE and Brss collections are
      * grouped by their respective time stamps and sorted by latitude.
      *
      * Finally the sorted FTE and Brss collections for each rotation are zipped
      * and only the patterns within the latitude limit are retained.
      *
      * Then the collections for each time stamp are loaded into Breeze Dense Vectors.
      *
      */
    val processed_data =
      carrington_rotation_table
        .join(data)
        .flatMap(
          pipes.process_timestamps_rotation2 > duplicate(pipes.sort_data) > pipes.zip_fte_brss
        )
        .filter(DataPipe(_._2.length == 360))
        .map(pipes.crop_data_by_latitude(latitude_limit))
        .map(pipes.load_slice_to_bdv)
        .to_zip(identityPipe[(DateTime, DenseVector[Double])])

    /**
      * The processed data set is now interpolated to ensure hourly cadence.
      * If specified by the user, a time history of the inputs is also constructed.
      */
    println("Interpolating FTE values to fill hourly cadence requirement")
    val interpolated_fte = dtfdata.dataset(
      processed_data.data
        .sliding(2)
        .filter(p => new Duration(p.head._1, p.last._1).getStandardHours > 1)
        .flatMap(i => {
          val duration  = new Duration(i.head._1, i.last._1).getStandardHours
          val delta_fte = (i.last._2 - i.head._2) / duration.toDouble

          (1 until duration.toInt)
            .map(
              l => (i.head._1.plusHours(l), i.head._2 + delta_fte * l.toDouble)
            )
        })
        .toIterable
    )

    //Transformations which load time history of the input features.
    val load_history = (history: Iterable[(DateTime, DenseVector[Double])]) => {

      val history_size = history.toSeq.length / fte_step

      val hs = history
        .map(_._2)
        .toSeq
        .zipWithIndex
        .filter(_._2 % fte_step == 0)
        .map(_._1)

      (
        history.last._1,
        DenseVector.vertcat(hs: _*)
      )
    }

    val generate_history = DataPipe(
      (s: Iterable[(DateTime, DenseVector[Double])]) =>
        if (deltaTFTE > 0)
          s.sliding((deltaTFTE * fte_step) + 1).map(load_history).toIterable
        else s
    )

    processed_data
      .concatenate(interpolated_fte)
      .transform(
        DataPipe[
          Iterable[(DateTime, DenseVector[Double])],
          Iterable[(DateTime, DenseVector[Double])]
        ](_.toSeq.sortBy(_._1))
      )
      .transform(generate_history)
      .to_zip(identityPipe[(DateTime, DenseVector[Double])])

  }

  /**
    * Load the OMNI solar wind time series as a [[Tensor]]
    *
    * @param start Starting time of the data.
    * @param end End time of the data.
    * @param deltaT The time window (t + l, t + l + h)
    * @param log_flag If set to true, log scale the velocity values.
    * @param quantity An integer column index corresponding to the OMNI
    *                 quantity to extract. Defaults to [[OMNIData.Quantities.V_SW]]
    *
    * @return A [[ZipDataSet]] with time indexed tensors containing
    *         sliding time histories of the solar wind.
    * */
  def load_solar_wind_data_bdv(
    start: DateTime,
    end: DateTime
  )(deltaT: (Int, Int),
    log_flag: Boolean,
    quantity: Int = OMNIData.Quantities.V_SW,
    ts_transform: DataPipe[Seq[Double], Seq[Double]] = identityPipe[Seq[Double]]
  ): ZipDataSet[DateTime, DenseVector[Double]] = {

    val transform: DataPipe[Seq[Double], Seq[Double]] = if (log_flag) {
      ts_transform > DataPipe((xs: Seq[Double]) => xs.map(math.log))
    } else {
      ts_transform
    }

    val omni_processing =
      OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(quantity) >
        IterableDataPipe(
          (p: (DateTime, Seq[Double])) =>
            p._1.isAfter(start) && p._1.isBefore(end)
        ) >
        IterableDataPipe(
          identityPipe[DateTime] * transform
        )

    val omni_data_path = pwd / 'data

    dtfdata
      .dataset(start.getYear to end.getYear)
      .map(
        DataPipe(
          (i: Int) =>
            omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
        )
      )
      .transform(omni_processing)
      .to_zip(
        identityPipe[DateTime] * DataPipe[Seq[Double], DenseVector[Double]](
          p => DenseVector(p.toArray)
        )
      )

  }

  /**
    * Load the OMNI solar wind time series as a [[Tensor]]
    *
    * @param start Starting time of the data.
    * @param end End time of the data.
    * @param deltaT The time window (t + l, t + l + h)
    * @param log_flag If set to true, log scale the velocity values.
    * @param quantity An integer column index corresponding to the OMNI
    *                 quantity to extract. Defaults to [[OMNIData.Quantities.V_SW]]
    *
    * @return A [[ZipDataSet]] with time indexed tensors containing
    *         sliding time histories of the solar wind.
    * */
  def load_solar_wind_data_bdv2(
    start: DateTime,
    end: DateTime
  )(deltaT: (Int, Int),
    log_flag: Boolean,
    quantity: Int = OMNIData.Quantities.V_SW,
    ts_transform: DataPipe[Seq[Double], Seq[Double]] = identityPipe[Seq[Double]]
  ): ZipDataSet[DateTime, (DenseVector[Double], DenseVector[Double])] = {

    val transform: DataPipe[Seq[Double], Seq[Double]] = if (log_flag) {
      ts_transform > DataPipe((xs: Seq[Double]) => xs.map(math.log))
    } else {
      ts_transform
    }

    val omni_processing =
      OMNILoader.biDirectionalWindow((27 * 24 - deltaT._1, deltaT._2), deltaT)(
        quantity
      ) >
        IterableDataPipe(
          (p: (DateTime, (Seq[Double], Seq[Double]))) =>
            p._1.isAfter(start) && p._1.isBefore(end)
        ) >
        IterableDataPipe(
          identityPipe[DateTime] * duplicate(transform)
        )

    val omni_data_path = pwd / 'data

    val load_into_bdv = DataPipe[Seq[Double], DenseVector[Double]](
      p => DenseVector(p.toArray)
    )

    dtfdata
      .dataset(start.getYear to end.getYear)
      .map(
        DataPipe(
          (i: Int) =>
            omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
        )
      )
      .transform(omni_processing)
      .to_zip(
        identityPipe[DateTime] * duplicate(load_into_bdv)
      )

  }

  def write_data_set[Input, Output](
    identifier: String,
    dataset: helios.data.DATA[Input, Output],
    input_to_seq: DataPipe[Input, Seq[Double]],
    output_to_seq: DataPipe[Output, Seq[Double]],
    directory: Path
  ): Unit = {

    val pattern_to_map =
      DataPipe[(DateTime, (Input, Output)), JValue](
        p =>
          (
            ("timestamp" -> p._1.toString("yyyy-MM-dd'T'HH:mm:ss'Z'")) ~
              ("targets" -> output_to_seq.run(p._2._2).toList) ~
              ("inputs"  -> input_to_seq.run(p._2._1))
          )
      )

    val map_to_json = DataPipe[JValue, String](p => write_json(p))

    val process_pattern = pattern_to_map > map_to_json

    val write_pattern_train: String => Unit =
      line =>
        write.append(
          directory / s"training_data_${identifier}.json",
          s"${line}\n"
        )

    val write_pattern_test: String => Unit =
      line =>
        write.append(
          directory / s"test_data_${identifier}.json",
          s"${line}\n"
        )

    dataset.training_dataset
      .map(process_pattern)
      .data
      .foreach(write_pattern_train)

    dataset.test_dataset
      .map(process_pattern)
      .data
      .foreach(write_pattern_test)
  }

  def read_data_set[T, U](
    training_data_file: Path,
    test_data_file: Path,
    load_input_pattern: DataPipe[Array[Double], T],
    load_output_pattern: DataPipe[Array[Double], U]
  ): helios.data.DATA[T, U] = {

    require(
      exists ! training_data_file && exists ! test_data_file,
      "Both training and test files must exist."
    )

    val dt_format = DateTimeFormat.forPattern("yyyy-MM-dd'T'HH:mm:ss'Z'")

    val read_file = DataPipe((p: Path) => read.lines ! p)

    val filter_non_empty_lines = IterableDataPipe((l: String) => !l.isEmpty)

    val read_json_record = IterableDataPipe((s: String) => parse(s))

    val load_record = IterableDataPipe((record: JValue) => {
      val dt = dt_format.parseDateTime(
        record
          .findField(p => p._1 == "timestamp")
          .get
          ._2
          .values
          .asInstanceOf[String]
      )
      val features = load_input_pattern(
        record
          .findField(p => p._1 == "inputs")
          .get
          ._2
          .values
          .asInstanceOf[List[Double]]
          .toArray
      )

      val targets = load_output_pattern(
        record
          .findField(p => p._1 == "targets")
          .get
          ._2
          .values
          .asInstanceOf[List[Double]]
          .toArray
      )
      (dt, (features, targets))
    })

    val pipeline = read_file > filter_non_empty_lines > read_json_record > load_record

    TFDataSet(
      dtfdata
        .dataset(pipeline(training_data_file))
        .to_zip(
          identityPipe[(DateTime, (T, U))]
        ),
      dtfdata
        .dataset(pipeline(test_data_file))
        .to_zip(identityPipe[(DateTime, (T, U))])
    )
  }

  def fte_model_preds(
    preds: Path,
    probs: Path,
    fte_data: Path
  ): timelag.utils.DataTriple = {

    val read_file = DataPipe((p: Path) => read.lines ! p)

    val split_lines = IterableDataPipe(
      (line: String) => line.split(',').map(_.toDouble)
    )

    val load_into_tensor = IterableDataPipe(
      (ls: Array[Double]) => dtf.tensor_f64(ls.length)(ls.toSeq: _*)
    )

    val filter_non_empty_lines = IterableDataPipe((l: String) => !l.isEmpty)

    val read_json_record = IterableDataPipe((s: String) => parse(s))

    val load_targets = IterableDataPipe((record: JValue) => {
      val targets_seq = record
        .findField(p => p._1 == "targets")
        .get
        ._2
        .values
        .asInstanceOf[List[Double]]

      val targets = dtf.tensor_f64(targets_seq.length)(targets_seq: _*)

      targets
    })

    val pipeline_fte = read_file > filter_non_empty_lines > read_json_record > load_targets

    val pipeline_model = read_file > split_lines > load_into_tensor

    (
      dtfdata.dataset(Seq(preds)).flatMap(pipeline_model),
      dtfdata.dataset(Seq(probs)).flatMap(pipeline_model),
      dtfdata.dataset(Seq(fte_data)).flatMap(pipeline_fte)
    )

  }

  def generate_dataset(
    fte_data_path: Path,
    c: FteOmniConfig,
    ts_transform_output: DataPipe[Seq[Double], Seq[Double]],
    tt_partition: DataPipe[(DateTime, (DenseVector[Double], DenseVector[
          Double
        ])), Boolean],
    conv_flag: Boolean = false
  ): helios.data.DATA[DenseVector[Double], DenseVector[Double]] = {

    val FteOmniConfig(
      FTEConfig(
        (start_year, end_year),
        deltaTFTE,
        fteStep,
        latitude_limit,
        log_scale_fte
      ),
      OMNIConfig(deltaT, log_scale_omni, quantity, use_persistence),
      multi_output,
      probabilistic_time_lags,
      timelag_prediction,
      fraction_variance
    ) = c

    val (start, end) = (
      new DateTime(start_year, 1, 1, 0, 0),
      new DateTime(end_year, 12, 31, 23, 59)
    )

    println("\nProcessing FTE Data")
    val fte_data = load_fte_data_bdv(
      fte_data_path,
      carrington_rotations,
      log_scale_fte,
      start,
      end
    )(deltaTFTE, fteStep, latitude_limit, conv_flag)

    println("Processing OMNI solar wind data")
    val omni_data = if (use_persistence) {
      println("Using 27 day persistence features")
      Left(
        load_solar_wind_data_bdv2(start, end)(
          deltaT,
          log_scale_omni,
          quantity,
          ts_transform_output
        )
      )
    } else {
      Right(
        load_solar_wind_data_bdv(start, end)(
          deltaT,
          log_scale_omni,
          quantity,
          ts_transform_output
        )
      )
    }

    println("Constructing joined data set")

    omni_data match {
      case Right(omni) => fte_data.join(omni).partition(tt_partition)
      case Left(omni) =>
        fte_data
          .join(omni)
          .map(
            identityPipe[DateTime] * DataPipe(
              (p: (
                DenseVector[Double],
                (DenseVector[Double], DenseVector[Double])
              )) => (DenseVector.vertcat(p._1, p._2._1), p._2._2)
            )
          )
          .partition(tt_partition)
    }

  }

  def _config_match(
    tf_summary_dir: Path,
    experiment_config: FteOmniConfig
  ): Boolean = {
    val existing_config = read_exp_config(tf_summary_dir / "config.json")

    val use_cached_config: Boolean = existing_config match {
      case None    => false
      case Some(c) => c == experiment_config
    }

    use_cached_config
  }

  def _dataset_serialized(tf_summary_dir: Path): Boolean = {

    val training_data_files = ls ! tf_summary_dir |? (_.segments.last
      .contains("training_data_"))
    val test_data_files = ls ! tf_summary_dir |? (_.segments.last
      .contains("test_data_"))

    training_data_files.length > 0 && test_data_files.length > 0

  }

  def setup_exp_data(
    year_range: Range = 2011 to 2017,
    test_year: Int = 2015,
    sw_threshold: Double = 700d,
    quantity: Int = OMNIData.Quantities.V_SW,
    ts_transform_output: DataPipe[Seq[Double], Seq[Double]] =
      identityPipe[Seq[Double]],
    deltaT: (Int, Int) = (48, 72),
    deltaTFTE: Int = 5,
    fteStep: Int = 1,
    latitude_limit: Double = 40d,
    fraction_pca: Double = 0.8,
    log_scale_fte: Boolean = false,
    log_scale_omni: Boolean = false,
    conv_flag: Boolean = false,
    fte_data_path: Path = home / 'Downloads / 'fte,
    summary_top_dir: Path = home / 'tmp,
    existing_exp: Option[Path] = None
  ): (FteOmniConfig, Path) = {
    val mo_flag: Boolean       = true
    val prob_timelags: Boolean = true

    val urv = UniformRV(0d, 1d)

    val sum_dir_prefix = if (conv_flag) "fte_omni_conv" else "fte_omni"

    val dt = DateTime.now()

    val summary_dir_index = {
      if (mo_flag) sum_dir_prefix + "_mo_tl_" + dt.toString("YYYY-MM-dd-HH-mm")
      else sum_dir_prefix + "_tl_" + dt.toString("YYYY-MM-dd-HH-mm")
    }

    val tf_summary_dir_tmp =
      existing_exp.getOrElse(summary_top_dir / summary_dir_index)

    val adj_fraction_pca = 1d

    val experiment_config = FteOmniConfig(
      FTEConfig(
        (year_range.min, year_range.max),
        deltaTFTE,
        fteStep,
        latitude_limit,
        log_scale_fte
      ),
      OMNIConfig(deltaT, log_scale_omni, quantity),
      multi_output = true,
      probabilistic_time_lags = true,
      timelag_prediction = "mode",
      fraction_variance = adj_fraction_pca
    )

    val existing_config = read_exp_config(tf_summary_dir_tmp / "config.json")

    val use_cached_config: Boolean =
      _config_match(tf_summary_dir_tmp, experiment_config)

    val tf_summary_dir = if (use_cached_config) {
      println("Using provided experiment directory to continue experiment")
      tf_summary_dir_tmp
    } else {
      println(
        "Ignoring provided experiment directory and starting fresh experiment"
      )

      write_exp_config(experiment_config, summary_top_dir / summary_dir_index)

      summary_top_dir / summary_dir_index
    }

    val use_cached_data = if (use_cached_config) {
      _dataset_serialized(tf_summary_dir)
    } else {
      false
    }

    val (test_start, test_end) = (
      new DateTime(test_year, 1, 1, 0, 0),
      new DateTime(test_year, 12, 31, 23, 59)
    )

    val tt_partition = DataPipe(
      (p: (DateTime, (DenseVector[Double], DenseVector[Double]))) =>
        if (p._1.isAfter(test_start) && p._1.isBefore(test_end))
          false
        else
          true
    )

    if (!use_cached_data) {

      val dataset =
        generate_dataset(
          fte_data_path,
          experiment_config,
          ts_transform_output,
          tt_partition,
          conv_flag
        )

      println("Serializing data sets")
      write_data_set(
        dt.toString("YYYY-MM-dd-HH-mm"),
        dataset,
        DataPipe[DenseVector[Double], Seq[Double]](_.toArray.toSeq),
        DataPipe[DenseVector[Double], Seq[Double]](_.toArray.toSeq),
        tf_summary_dir
      )
    }

    (experiment_config, tf_summary_dir)
  }

}
