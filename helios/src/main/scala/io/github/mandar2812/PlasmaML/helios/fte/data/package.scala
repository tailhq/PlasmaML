package io.github.mandar2812.PlasmaML.helios.fte

import ammonite.ops._
import org.joda.time._
import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.tensorflow.data._
import io.github.mandar2812.dynaml.tensorflow.utils._
import io.github.mandar2812.dynaml.tensorflow.{dtf, dtfdata, dtflearn, dtfutils}
import io.github.mandar2812.dynaml.tensorflow.implicits._
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.DynaMLPipe._
import _root_.io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import org.platanios.tensorflow.api._

package object data {

  //Set time zone to UTC
  DateTimeZone.setDefault(DateTimeZone.UTC)

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

  val fte_file = MetaPipe(
    (data_path: Path) =>
      (carrington_rotation: Int) =>
        data_path / s"HMIfootpoint_ch_csss${carrington_rotation}HR.dat"
  )

  case class FTEPattern(data: (Double, Double, Option[Double])) extends AnyVal {

    def _1: Double         = data._1
    def _2: Double         = data._2
    def _3: Option[Double] = data._3
  }

  //type FTEPattern = FTEPattern

  val process_fte_file = {
    fte_file >> (
      DataPipe((p: Path) => (read.lines ! p).toStream) >
        Seq.fill(4)(dropHead).reduceLeft(_ > _) >
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

  case class FTEConfig()

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

  val process_rotation = DataPipe(
    (rotation_data: (Int, (CarringtonRotation, Iterable[FTEPattern]))) => {

      val (_, (rotation, fte)) = rotation_data

      val duration = new Duration(rotation.start, rotation.end)

      val time_jump = duration.getMillis / 360.0

      fte.map(
        p =>
          (
            rotation.end.toInstant
              .minus((time_jump * p._1).toLong)
              .toDateTime,
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
      //Tensor[Double](s.map(_._3.get).map(log_transformation))
      //.reshape(Shape(s.length))
    )

    val sort_by_date = DataPipe[Iterable[(DateTime, Seq[FTEPattern])], Iterable[
      (DateTime, Seq[FTEPattern])
    ]](
      _.toSeq.sortBy(_._1)
    )

    val processed_fte_data = {
      fte_data
        .flatMap(process_rotation)
        .transform(
          DataPipe[Iterable[(DateTime, FTEPattern)], Iterable[
            (DateTime, Seq[FTEPattern])
          ]](
            _.groupBy(_._1).map(p => (p._1, p._2.map(_._2).toSeq.sortBy(_._2)))
          )
        )
        .filter(DataPipe(_._2.length == 180))
        .map(crop_data_by_latitude)
        .map(image_dt_roundoff * identityPipe[Seq[FTEPattern]])
        .transform(sort_by_date)
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
          s.sliding((deltaTFTE + 1) * fte_step).map(load_history).toIterable
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

    val processed_fte_data = {
      fte_data
        .flatMap(process_rotation)
        .transform(
          DataPipe[Iterable[(DateTime, FTEPattern)], Iterable[
            (DateTime, Seq[FTEPattern])
          ]](
            _.groupBy(_._1).map(p => (p._1, p._2.map(_._2).toSeq.sortBy(_._2)))
          )
        )
        .filter(DataPipe(_._2.length == 180))
        .map(crop_data_by_latitude)
        .map(image_dt_roundoff * identityPipe[Seq[FTEPattern]])
        .transform(sort_by_date)
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
          s.sliding((deltaTFTE + 1) * fte_step).map(load_history).toIterable
        else s
    )

    processed_fte_data
      .concatenate(interpolated_fte)
      .transform(
        DataPipe[Iterable[(DateTime, DenseVector[Double])], Iterable[
          (DateTime, DenseVector[Double])
        ]](_.toSeq.sortBy(_._1).toIterable)
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

  val load_bdv_to_tf = identityPipe[DateTime] * DataPipe[DenseVector[Double], Tensor[
    Double
  ]](p => dtf.tensor_f64(p.size)(p.toArray.toSeq: _*))

}
