package io.github.mandar2812.PlasmaML.helios.fte

import ammonite.ops._
import org.joda.time._
import org.joda.time.format.DateTimeFormat
import breeze.linalg.{DenseVector, DenseMatrix}
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

  case class OMNIConfig(deltaT: (Int, Int), log_flag: Boolean)

  case class FteOmniConfig(
    fte_config: FTEConfig,
    omni_config: OMNIConfig,
    multi_output: Boolean = true,
    probabilistic_time_lags: Boolean = true,
    timelag_prediction: String = "mode",
    fraction_variance: Double = 1d)
      extends helios.Config

  //Load the Carrington Rotation Table
  val carrington_rotation_table: Path = pwd / 'data / "CR_Table.rdb"

  val process_carrington_file: DataPipe[Path, Iterable[Array[String]]] =
    DataPipe((p: Path) => (read.lines ! p).toStream) >
      dropHead >
      dropHead >
      trimLines >
      replaceWhiteSpaces >
      splitLine

  case class CarringtonRotation(start: DateTime, end: DateTime) {

    def contains(dt: DateTime): Boolean = dt.isAfter(start) && dt.isBefore(end)
  }

  val read_time_stamps = DataPipe((s: Array[String]) => {

    val datetime_pattern = "YYYY.MM.dd_HH:mm:ss"
    val dt               = format.DateTimeFormat.forPattern(datetime_pattern)

    val limits = (DateTime.parse(s(1), dt), DateTime.parse(s(3), dt))

    (s.head.toInt, CarringtonRotation(limits._1, limits._2))
  })

  val carrington_rotations: ZipDataSet[Int, CarringtonRotation] =
    dtfdata
      .dataset(process_carrington_file(carrington_rotation_table))
      .to_zip(read_time_stamps)


  val read_lines_gong = (gong_file: Path) => (read.lines ! gong_file).toIterable.drop(3)

  val read_lines_hmi = (hmi_file: Path) => (read.lines ! hmi_file).toIterable.drop(4) 
  
  val fte_file = MetaPipe(
    (data_path: Path) =>
      (carrington_rotation: Int) => {
        val hmi_file  = data_path / s"HMIfootpoint_ch_csss${carrington_rotation}HR.dat"
        val gong_file = data_path / s"GONGfootpoint_ch_csss${carrington_rotation}HR.txt"
        
        if(exists ! hmi_file) read_lines_hmi(hmi_file) 
        else read_lines_gong(gong_file)
      }
  )

  case class FTEPattern(data: (Double, Double, Option[Double])) extends AnyVal {

    def _1: Double         = data._1
    def _2: Double         = data._2
    def _3: Option[Double] = data._3
  }

  //type FTEPattern = FTEPattern

  val process_fte_file = {
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

          FTEPattern(lon, lat, fte)

        })
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
  ): Iterable[(Int, Iterable[FTEPattern])] =
    try {
      Iterable((cr, process_fte_file(data_path)(cr)))
    } catch {
      case _: java.nio.file.NoSuchFileException => Iterable()
    }

  val process_rotation
    : DataPipe[(Int, (CarringtonRotation, Iterable[FTEPattern])), Iterable[
      (DateTime, FTEPattern)
    ]] = DataPipe(
    (rotation_data) => {

      val (_, (rotation, fte)) = rotation_data

      val duration = new Duration(rotation.start, rotation.end)

      val time_jump = duration.getMillis / 360.0

      val time_stamp = (p: FTEPattern) =>
        rotation.end.toInstant
          .minus((time_jump * p._1).toLong)
          .toDateTime

      fte.map(
        p =>
          (
            time_stamp(p),
            p
          )
      )

    }
  )

  val image_dt_roundoff: DataPipe[DateTime, DateTime] = DataPipe(
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

    val clamp_fte: FTEPattern => FTEPattern = (p: FTEPattern) =>
      p._3 match {
        case Some(f) =>
          if (math.abs(f) <= 1000d) p
          else FTEPattern(p._1, p._2, Some(1000d * math.signum(f)))
        case None => p
      }

    val load_fte_for_rotation = DataPipe(get_fte_for_rotation(data_path) _)

    val fte = dtfdata
      .dataset(start_rotation to end_rotation)
      .flatMap(load_fte_for_rotation)
      .map(
        identityPipe[Int] * IterableDataPipe[FTEPattern, FTEPattern](clamp_fte)
      )
      .to_zip(identityPipe[(Int, Iterable[FTEPattern])])

    val fte_data = carrington_rotation_table.join(fte)

    val log_transformation =
      (x: Double) =>
        if (log_flag) {
          if (math.abs(x) < 1d) 0d
          else math.log10(math.abs(x))
        } else x

    val crop_data_by_latitude = DataPipe(
      (pattern: (DateTime, Seq[FTEPattern])) =>
        (
          pattern._1,
          pattern._2.filter(ftep => math.abs(ftep._2) <= latitude_limit)
        )
    )

    val load_slice_to_tensor = DataPipe[Seq[FTEPattern], Tensor[Double]](
      (s: Seq[FTEPattern]) =>
        dtf.tensor_f64(s.length)(s.map(_._3.get).map(log_transformation): _*)
    )

    val sort_by_date = DataPipe[Iterable[(DateTime, Seq[FTEPattern])], Iterable[
      (DateTime, Seq[FTEPattern])
    ]](
      _.toSeq.sortBy(_._1)
    )

    val sort_by_latitude = DataPipe[Iterable[(DateTime, FTEPattern)], Iterable[
      (DateTime, Seq[FTEPattern])
    ]](
      _.groupBy(_._1).map(p => (p._1, p._2.map(_._2).toSeq.sortBy(_._2)))
    )

    val processed_fte_data = {
      fte_data
        .flatMap(
          process_rotation >
            IterableDataPipe(image_dt_roundoff * identityPipe[FTEPattern]) >
            sort_by_latitude >
            sort_by_date
        )
        .filter(DataPipe(_._2.length == 180))
        .map(crop_data_by_latitude)
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
    quantity: Int = OMNIData.Quantities.V_SW
  ): ZipDataSet[DateTime, Tensor[Double]] = {

    val omni_processing =
      OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(quantity) >
        IterableDataPipe(
          (p: (DateTime, Seq[Double])) =>
            p._1.isAfter(start) && p._1.isBefore(end)
        ) >
        IterableDataPipe(
          (p: (DateTime, Seq[Double])) =>
            (p._1, if (log_flag) p._2.map(math.log) else p._2)
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

      val perform_lossy_pca =
        calculatePCAScalesFeatures(false) >
          tup2_2[Iterable[DenseVector[Double]], PCAScaler] >
          compressPCA(fraction)

      val scale_features =
        DataPipe[
          Iterable[DenseVector[Double]],
          (Iterable[DenseVector[Double]], GaussianScaler)
        ](ds => {
          val (mean, variance) = dutils.getStats(ds)
          val gs               = GaussianScaler(mean, variance)

          (ds.map(gs(_)), gs)
        }) >
          (perform_lossy_pca * identityPipe[GaussianScaler]) >
          DataPipe2[CompressedPCAScaler, GaussianScaler, Scaler[
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

  /**
    * Creates a DynaML data set consisting of time FTE values.
    * The FTE values are loaded in a [[Tensor]] object.
    * */
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

    val start_rotation =
      carrington_rotation_table.filter(_._2.contains(start)).data.head._1

    val end_rotation =
      carrington_rotation_table.filter(_._2.contains(end)).data.head._1

    val clamp_fte: FTEPattern => FTEPattern = (p: FTEPattern) =>
      p._3 match {
        case Some(f) =>
          if (math.abs(f) <= 1000d) p
          else FTEPattern(p._1, p._2, Some(1000d * math.signum(f)))
        case None => p
      }

    val load_fte_for_rotation = DataPipe(get_fte_for_rotation(data_path) _)

    val fte = dtfdata
      .dataset(start_rotation to end_rotation)
      .flatMap(load_fte_for_rotation)
      .map(
        identityPipe[Int] * IterableDataPipe[FTEPattern, FTEPattern](clamp_fte)
      )
      .to_zip(identityPipe[(Int, Iterable[FTEPattern])])

    val fte_data = carrington_rotation_table.join(fte)

    val log_transformation =
      (x: Double) =>
        if (log_flag) {
          if (math.abs(x) < 1d) 0d
          else math.log10(math.abs(x))
        } else x

    val crop_data_by_latitude = DataPipe(
      (pattern: (DateTime, Seq[FTEPattern])) =>
        (
          pattern._1,
          pattern._2.filter(ftep => math.abs(ftep._2) <= latitude_limit)
        )
    )

    val load_slice_to_tensor = DataPipe[Seq[FTEPattern], DenseVector[Double]](
      (s: Seq[FTEPattern]) =>
        DenseVector(s.map(_._3.get).map(log_transformation).toArray)
    )

    val sort_by_date = DataPipe[Iterable[(DateTime, Seq[FTEPattern])], Iterable[
      (DateTime, Seq[FTEPattern])
    ]](
      _.toSeq.sortBy(_._1)
    )

    val sort_by_latitude = DataPipe[Iterable[(DateTime, FTEPattern)], Iterable[
      (DateTime, Seq[FTEPattern])
    ]](
      _.groupBy(_._1).map(p => (p._1, p._2.map(_._2).toSeq.sortBy(_._2)))
    )

    val processed_fte_data = {
      fte_data
        .flatMap(
          process_rotation >
            IterableDataPipe(image_dt_roundoff * identityPipe[FTEPattern]) >
            sort_by_latitude >
            sort_by_date
        )
        .filter(DataPipe(_._2.length == 180))
        .map(crop_data_by_latitude)
        .map(identityPipe[DateTime] * load_slice_to_tensor)
        .to_zip(identityPipe[(DateTime, DenseVector[Double])])
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
            .map(
              l => (i.head._1.plusHours(l), i.head._2 + delta_fte * l.toDouble)
            )
        })
        .toIterable
    )

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

    processed_fte_data
      .concatenate(interpolated_fte)
      .transform(
        DataPipe[Iterable[(DateTime, DenseVector[Double])], Iterable[
          (DateTime, DenseVector[Double])
        ]](_.toSeq.sortBy(_._1))
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
    quantity: Int = OMNIData.Quantities.V_SW
  ): ZipDataSet[DateTime, DenseVector[Double]] = {

    val omni_processing =
      OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(quantity) >
        IterableDataPipe(
          (p: (DateTime, Seq[Double])) =>
            p._1.isAfter(start) && p._1.isBefore(end)
        ) >
        IterableDataPipe(
          (p: (DateTime, Seq[Double])) =>
            (p._1, if (log_flag) p._2.map(math.log) else p._2)
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

  def write_exp_config(config: FteOmniConfig, dir: Path): Unit = {
    if (!(exists ! dir / "config.json")) {
      val config_json = write_json(config)
      write(dir / "config.json", config_json)
    }
  }

  def write_fte_data_set[Input](
    identifier: String,
    dataset: helios.data.TF_DATA_T2[Input, Double],
    input_to_seq: DataPipe[Input, Seq[Double]],
    directory: Path
  ): Unit = {

    val pattern_to_map =
      DataPipe[(DateTime, (Input, Tensor[Double])), JValue](
        p =>
          (
            ("timestamp" -> p._1.toString("yyyy-MM-dd'T'HH:mm:ss'Z'")) ~
              ("targets" -> dtfutils.toDoubleSeq(p._2._2).toList) ~
              ("fte"     -> input_to_seq.run(p._2._1))
          )
      )

    val map_to_json = DataPipe[JValue, String](p => write_json(p))

    val process_pattern = pattern_to_map > map_to_json

    val write_pattern_train: String => Unit = 
      line => write.append(
        directory / s"training_data_${identifier}.json",
        s"${line}\n"
      )

    val write_pattern_test: String => Unit = 
      line => write.append(
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

  def read_exp_config(
    file: Path
  ): Option[FteOmniConfig] = 
    if(exists! file) {
      try {
        val config = parse((read.lines! file).head).values.asInstanceOf[Map[String, Any]]
        val fte_config = config("fte_config").asInstanceOf[Map[String, Any]]
        val omni_config = config("omni_config").asInstanceOf[Map[String, Any]]

        val omni_deltaT =  omni_config("deltaT").asInstanceOf[Map[String, BigInt]]
        val fte_data_limits =  fte_config("data_limits").asInstanceOf[Map[String, BigInt]]

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
            omni_config("log_flag").asInstanceOf[Boolean]
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

  def read_data_set(
    training_data_file: Path,
    test_data_file: Path
  ): helios.data.TF_DATA_T2[DenseVector[Double], Double] = {

    require(
      exists! training_data_file && exists! test_data_file, 
      "Both training and test files must exist.")

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
      val features = DenseVector(
        record
          .findField(p => p._1 == "fte")
          .get
          ._2
          .values
          .asInstanceOf[List[Double]]
          .toArray
      )

      val targets_seq = record
        .findField(p => p._1 == "targets")
        .get
        ._2
        .values
        .asInstanceOf[List[Double]]

      val targets = dtf.tensor_f64(targets_seq.length)(targets_seq: _*)

      (dt, (features, targets))
    })

    val pipeline = read_file > filter_non_empty_lines > read_json_record > load_record

    TFDataSet(
      dtfdata
        .dataset(pipeline(training_data_file))
        .to_zip(
          identityPipe[(DateTime, (DenseVector[Double], Tensor[Double]))]
        ),
      dtfdata
        .dataset(pipeline(test_data_file))
        .to_zip(identityPipe[(DateTime, (DenseVector[Double], Tensor[Double]))])
    )
  }

}
