package io.github.mandar2812.PlasmaML.helios

import java.io.IOException
import java.nio.file.Files

import ammonite.ops._
import breeze.linalg.DenseVector
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.filter.GrayscaleFilter
import io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.data.SDOData.Instruments._
import io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.probability.{DiscreteDistrRV, MultinomialRV}
import io.github.mandar2812.dynaml.tensorflow.{dtf, dtfdata, dtfpipe}
import io.github.mandar2812.dynaml.tensorflow.data.{
  AbstractDataSet,
  DataSet,
  TFDataSet,
  ZipDataSet
}
import io.github.mandar2812.dynaml.tensorflow.utils.{
  GaussianScalerTF,
  MinMaxScalerTF
}
import io.github.mandar2812.dynaml.DynaMLPipe.{identityPipe => id, _}
import io.github.mandar2812.dynaml.utils
import org.joda.time._
import org.platanios.tensorflow.api._
import _root_.org.json4s._
import _root_.org.json4s.JsonDSL._
import _root_.org.json4s.jackson.Serialization.{
  read => read_json,
  write => write_json
}

/**
  * <h3>Helios Data Facility</h3>
  *
  * Package enabling downloading and storage and processing
  * of heliospheric image data.
  *
  * @author mandar2812
  * */
package object data {

  implicit val formats = DefaultFormats + FieldSerializer[Map[String, Any]]()

  private implicit val dateOrdering = new Ordering[DateTime] {
    override def compare(x: DateTime, y: DateTime): Int =
      if (x.isBefore(y)) -1 else 1
  }

  type IMAGE_PATTERN = (DateTime, Path)

  /**
    * A simple data pattern, consisting of
    * a time stamp, path to an image, and a sequence of numbers
    * */
  type PATTERN = (DateTime, (Seq[Path], Seq[Double]))

  /**
    * A pattern, consisting of
    * a time stamp, path to an image, a tuple of numeric sequences
    * */
  type PATTERN_EXT = (DateTime, (Path, (Seq[Double], Seq[Double])))

  /**
    * A pattern, consisting of
    * a time stamp, a collection of images from multiple sources,
    * and a sequence of numbers
    * */
  type MC_PATTERN = (DateTime, (Map[SolarImagesSource, Seq[Path]], Seq[Double]))

  /**
    * A pattern, consisting of
    * a time stamp, a collection of images from multiple sources,
    * and a tuple of sequence of numbers
    * */
  type MC_PATTERN_EXT =
    (DateTime, (Map[SOHO, Seq[Path]], (Seq[Double], Seq[Double])))

  type HELIOS_IMAGE_DATA       = DataSet[IMAGE_PATTERN]
  type HELIOS_OMNI_DATA        = DataSet[PATTERN]
  type HELIOS_MC_OMNI_DATA     = DataSet[MC_PATTERN]
  type HELIOS_OMNI_DATA_EXT    = DataSet[PATTERN_EXT]
  type HELIOS_MC_OMNI_DATA_EXT = Iterable[MC_PATTERN_EXT]

  type IMAGE_TS[T, U] = (Tensor[T], Tensor[U])

  type TF_IMAGE_DATA[T] = TFDataSet[Tensor[T]]

  type TF_DATA[T, U] = TFDataSet[(Tensor[T], Tensor[U])]

  type TF_DATA_T[T, U] = TFDataSet[(DateTime, (Tensor[T], Tensor[U]))]

  type TF_DATA_T2[T, U] = TFDataSet[(DateTime, (T, Tensor[U]))]

  type DATA[T, U] = TFDataSet[(DateTime, (T, U))]

  type TF_DATA_EXT[T, U] = TFDataSet[((Tensor[T], Tensor[U]), Tensor[U])]

  type TF_MC_DATA_EXT[T, U] = AbstractDataSet[(Tensor[T], Tensor[U]), Tensor[U]]

  type SC_TF_MC_DATA_EXT[T, U] = (
    TF_MC_DATA_EXT[T, U],
    (ReversibleScaler[(Tensor[T], Tensor[U])], MinMaxScalerTF[U])
  )

  type SC_TF_DATA_EXT[T, U] = (
    TF_DATA_EXT[T, U],
    (ReversibleScaler[(Tensor[T], Tensor[U])], MinMaxScalerTF[U])
  )

  type SC_TF_DATA[T, U] =
    (TF_DATA[T, U], (ReversibleScaler[Tensor[T]], MinMaxScalerTF[U]))

  type SC_TF_DATA_T[T, U] =
    (TF_DATA_T[T, U], (ReversibleScaler[Tensor[T]], MinMaxScalerTF[U]))

  type SC_TF_DATA_T2[T, U] =
    (TF_DATA_T2[T, U], (Scaler[T], GaussianScalerTF[U]))

  private def TF_DATA_EXT[T: TF, U: TF: IsFloatOrDouble](
    trData: IMAGE_TS[T, U],
    trLabels: Tensor[U],
    sizeTr: Int,
    tData: IMAGE_TS[T, U],
    tLabels: Tensor[U],
    sizeT: Int
  ): AbstractDataSet[IMAGE_TS[T, U], Tensor[U]] =
    AbstractDataSet(trData, trLabels, sizeTr, tData, tLabels, sizeT)

  /**
    * Download a resource (image, file) from a sequence of urls to a specified
    * disk location.
    * */
  def download_batch(path: Path)(urls: List[String]): Unit = {
    urls.par.foreach(
      s => utils.downloadURL(s, (path / s.split('/').last).toString())
    )
  }

  /**
    * Perform a bulk download of images within some date range
    * */
  def download_day_range(
    download: LocalDate => Unit
  )(start: LocalDate,
    end: LocalDate
  ): Unit = {

    val num_days = new Duration(
      start.toDateTimeAtStartOfDay,
      end.toDateTimeAtStartOfDay
    ).getStandardDays.toInt

    (0 to num_days).map(start.plusDays).par.foreach(download)
  }

  def download_month_range(
    download: YearMonth => Unit
  )(start: YearMonth,
    end: YearMonth
  ): Unit = {

    val period = new Period(
      start.toLocalDate(1).toDateTimeAtStartOfDay,
      end.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val num_months = (12 * period.getYears) + period.getMonths

    (0 to num_months).map(start.plusMonths).par.foreach(download)
  }

  sealed trait Source
  sealed trait SolarImagesSource extends Source {
    val instrument: String
    val size: Int
  }

  case class SOHO(
    override val instrument: String,
    override val size: Int = SOHOData.Resolutions.s512)
      extends SolarImagesSource {

    override def toString: String = "SOHO_" + instrument + "_" + size

  }

  case class SDO(
    override val instrument: String,
    override val size: Int = SDOData.Resolutions.s512)
      extends SolarImagesSource {

    override def toString: String = "SDO_" + instrument + "_" + size

  }

  case class GOES(
    quantity: String = GOESData.Quantities.XRAY_FLUX_5m,
    format: String = GOESData.Formats.CSV)
      extends Source

  case class OMNI(quantity: Int = OMNIData.Quantities.V_SW) extends Source {
    override def toString: String =
      "OMNI_" + OMNIData.columnNames(quantity).replace(" ", "_")
  }

  def image_process_metadata[S <: SolarImagesSource](
    image_source: S
  ): (DataPipe[Image, Image], Int, DataPipe[Image, Array[Byte]]) =
    image_source match {
      case _: SOHO =>
        (
          DataPipe((i: Image) => i.filter(GrayscaleFilter)),
          1,
          DataPipe((i: Image) => i.argb.map(_.last.toByte))
        )

      case SDO(AIA094335193, _) =>
        (
          id[Image],
          4,
          DataPipe((i: Image) => i.argb.flatten.map(_.toByte))
        )

      case SDO(HMI171, _) =>
        (
          id[Image],
          4,
          DataPipe((i: Image) => i.argb.flatten.map(_.toByte))
        )

      case _: SDO =>
        (
          DataPipe((i: Image) => i.filter(GrayscaleFilter)),
          1,
          DataPipe((i: Image) => i.argb.map(_.last.toByte))
        )
    }

  private var size_buffer = 500

  private var image_byte_buffer = 20

  def buffer_size_(s: Int) = size_buffer = s

  def _buffer_size: Int = size_buffer

  def image_buffer_size_(s: Int) = image_byte_buffer = s

  def _image_buffer_size = image_byte_buffer

  val read_image = DataPipe((p: Path) => Image.fromPath(p.toNIO))

  val non_corrupted_pattern =
    DataPipe[(Option[Tensor[_]], Seq[Double]), Boolean](_._1.isDefined)

  def get_image_tensor = DataPipe((c: Option[Tensor[UByte]]) => c.get)

  def available_bytes(p: Path): Int = {
    val in = Files.newInputStream(p.toNIO)

    val available_bytes: Int = try in.available()
    catch {
      case _: IOException => -1
      case _: Exception   => -1
    }

    in.close()

    available_bytes
  }

  def image_to_tensor: MetaPipe21[Int, Int, Array[Byte], Tensor[UByte]] =
    MetaPipe21[Int, Int, Array[Byte], Tensor[UByte]](
      (size: Int, channels: Int) =>
        (data: Array[Byte]) => {

          //Divide the image into sectors
          //Construct an image for each sector
          //Concatenate in column then row wise fashion
          tfi.concatenate[UByte](
            data.toIterable
              .grouped(channels)
              .grouped(size)
              .grouped(_image_buffer_size)
              .map(arr => {
                val byte_arr: Array[Byte] = arr.flatten.flatten.toArray
                dtf.tensor_from_buffer[UByte](
                  Shape(_image_buffer_size, size, channels),
                  byte_arr
                )
              })
              .toSeq,
            axis = 0
          )
        }
    )

  val image_scale: MetaPipe[Double, Image, Image] = MetaPipe(
    (factor: Double) => (image: Image) => image.copy.scale(factor)
  )

  val image_central_patch: MetaPipe21[Double, Int, Image, Image] =
    MetaPipe21(
      (image_magic_ratio: Double, image_sizes: Int) =>
        (image: Image) => {
          val start      = (1.0 - image_magic_ratio) * image_sizes / 2
          val patch_size = image_sizes * image_magic_ratio

          image.subimage(
            start.toInt,
            start.toInt,
            patch_size.toInt,
            patch_size.toInt
          )
        }
    )

  def get_patch_range(magic_ratio: Double, image_sizes: Int) = {
    val start      = (1.0 - magic_ratio) * image_sizes / 2
    val patch_size = image_sizes * magic_ratio

    start.toInt to (start.toInt + patch_size.toInt)
  }

  val extract_central_patch
    : MetaPipe21[Double, Int, Output[UByte], Output[UByte]] =
    MetaPipe21(
      (image_magic_ratio: Double, image_sizes: Int) =>
        (image: Output[UByte]) => {
          val start      = (1.0 - image_magic_ratio) * image_sizes / 2
          val patch_size = image_sizes * image_magic_ratio

          val patch_range = start.toInt to (start.toInt + patch_size.toInt)

          dtfpipe
            .extract_image_patch[UByte](patch_range, patch_range)
            .run(image)
        }
    )

  val image_pixel_scaler =
    MinMaxScalerTF[UByte](Tensor(UByte(0.toByte)), Tensor(UByte(255.toByte)))

  def std_images_and_outputs[
    T: TF: IsFloatOrDouble
  ]: DataPipe2[Tensor[UByte], Tensor[
    T
  ], ((Tensor[UByte], Tensor[T]), (MinMaxScalerTF[UByte], MinMaxScalerTF[T]))] =
    DataPipe2((features: Tensor[UByte], labels: Tensor[T]) => {

      val labels_min = labels.min(axes = 0)
      val labels_max = labels.max(axes = 0)

      val (features_scaler, labels_scaler) = (
        image_pixel_scaler,
        MinMaxScalerTF[T](labels_min, labels_max)
      )

      val (features_scaled, labels_scaled) = (
        features,
        labels_scaler(labels)
      )

      ((features_scaled, labels_scaled), (features_scaler, labels_scaler))
    })

  def gauss_std[
    T: TF: IsNotQuantized
  ]: DataPipe[Tensor[T], (Tensor[T], GaussianScalerTF[T])] =
    DataPipe((labels: Tensor[T]) => {

      val labels_mean = labels.mean(axes = 0)

      val n_data = labels.shape(0).scalar.toDouble

      val labels_sd =
        labels
          .subtract(labels_mean)
          .square
          .mean(axes = 0)
          .multiply(Tensor(n_data / (n_data - 1d)).castTo[T])
          .sqrt

      val labels_scaler = GaussianScalerTF[T](labels_mean, labels_sd)

      val labels_scaled = labels_scaler(labels)

      (labels_scaled, labels_scaler)
    })

  def minmax_std[
    T: TF: IsNotQuantized
  ]: DataPipe[Tensor[T], (Tensor[T], MinMaxScalerTF[T])] =
    DataPipe((labels: Tensor[T]) => {

      val labels_min = labels.min(axes = 0)
      val labels_max = labels.max(axes = 0)

      val labels_scaler = MinMaxScalerTF[T](labels_min, labels_max)

      val labels_scaled = labels_scaler(labels)

      (labels_scaled, labels_scaler)
    })

  protected def id_scaler[T: TF]: ReversibleScaler[Tensor[T]] =
    new ReversibleScaler[Tensor[T]] {
      override val i: Scaler[Tensor[T]] = Scaler((o: Tensor[T]) => o)

      override def run(data: Tensor[T]): Tensor[T] = data
    }

  def scale_helios_dataset[
    U: TF: IsFloatOrDouble
  ]: DataPipe[TF_DATA[UByte, U], SC_TF_DATA[UByte, U]] =
    DataPipe((dataset: TF_DATA[UByte, U]) => {

      val concat_targets = tfi.stack(
        dataset.training_dataset
          .map(tup2_2[Tensor[UByte], Tensor[U]])
          .data
          .toSeq
      )

      val (min, max) =
        (concat_targets.min(axes = 0), concat_targets.max(axes = 0))

      val targets_scaler = MinMaxScalerTF[U](min, max)

      (
        dataset.copy(
          training_dataset =
            dataset.training_dataset.map(id_scaler[UByte] * targets_scaler)
        ),
        (id_scaler[UByte], targets_scaler)
      )
    })

  def scale_timed_dataset[
    U: TF: IsFloatOrDouble
  ]: DataPipe[TF_DATA_T[UByte, U], SC_TF_DATA_T[UByte, U]] =
    DataPipe((dataset: TF_DATA_T[UByte, U]) => {

      val concat_targets =
        tfi.stack(
          dataset.training_dataset
            .map(
              tup2_2[DateTime, (Tensor[UByte], Tensor[U])] > tup2_2[Tensor[
                UByte
              ], Tensor[U]]
            )
            .data
            .toSeq
        )

      val (min, max) =
        (concat_targets.min(axes = 0), concat_targets.max(axes = 0))

      val targets_scaler = MinMaxScalerTF[U](min, max)

      val dt_sc = new ReversibleScaler[DateTime] {
        override val i: Scaler[DateTime] = Scaler((o: DateTime) => o)

        override def run(data: DateTime): DateTime = data
      }

      (
        dataset.copy(
          training_dataset = dataset.training_dataset
            .map(dt_sc * (id_scaler[UByte] * targets_scaler))
        ),
        (id_scaler[UByte], targets_scaler)
      )
    })

  def scale_helios_dataset_mc_ext[
    U: TF: IsFloatOrDouble
  ]: DataPipe[TF_MC_DATA_EXT[UByte, U], SC_TF_MC_DATA_EXT[UByte, U]] =
    DataPipe((dataset: TF_MC_DATA_EXT[UByte, U]) => {

      val (norm_tr_images_and_labels, scalers) =
        std_images_and_outputs.run(dataset.trainData._1, dataset.trainLabels)
      val (norm_histories, history_scaler) =
        minmax_std.run(dataset.trainData._2)

      val features_scaler = scalers._1 * history_scaler

      (
        dataset.copy(
          trainLabels = norm_tr_images_and_labels._2,
          trainData = (norm_tr_images_and_labels._1, norm_histories),
          testData = (dataset.testData._1, history_scaler(dataset.testData._2))
        ),
        (features_scaler, scalers._2)
      )
    })

  def scale_helios_dataset_ext[U: TF: IsFloatOrDouble] =
    DataPipe[TF_DATA_EXT[UByte, U], SC_TF_DATA_EXT[UByte, U]](dataset => {

      val concat_targets = tfi.stack(
        dataset.training_dataset
          .map(tup2_2[(Tensor[UByte], Tensor[U]), Tensor[U]])
          .data
          .toSeq
      )

      val concat_history = tfi.stack(
        dataset.training_dataset
          .map(
            tup2_1[(Tensor[UByte], Tensor[U]), Tensor[U]] > tup2_2[Tensor[
              UByte
            ], Tensor[U]]
          )
          .data
          .toSeq
      )

      val (min, max) =
        (concat_targets.min(axes = 0), concat_targets.max(axes = 0))
      val (min_h, max_h) =
        (concat_history.min(axes = 0), concat_history.max(axes = 0))

      val targets_scaler = MinMaxScalerTF[U](min, max)

      val history_scaler = MinMaxScalerTF[U](min_h, max_h)

      val inputs_scaler = id_scaler[UByte] * history_scaler

      (
        dataset.copy(
          training_dataset =
            dataset.training_dataset.map(inputs_scaler * targets_scaler)
        ),
        (inputs_scaler, targets_scaler)
      )

    })

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
    source: data.Source,
    download_path: Path
  )(start: LocalDate,
    end: LocalDate
  ): Unit = source match {

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
    source: data.Source,
    download_path: Path
  )(start: YearMonth,
    end: YearMonth
  ): Unit = source match {

    case data.GOES(quantity, format) =>
      GOESLoader.bulk_download(download_path)(quantity, format)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }

  def load_images[T <: SolarImagesSource](
    data_path: Path,
    year_month: YearMonth,
    image_source: T,
    dirTreeCreated: Boolean = true
  ): Iterable[(DateTime, Path)] =
    try {
      image_source match {
        case SOHO(i, s) =>
          SOHOLoader.load_images(
            data_path,
            year_month,
            SOHO(i, s),
            dirTreeCreated
          )
        case SDO(i, s) =>
          SDOLoader.load_images(
            data_path,
            year_month,
            SDO(i, s),
            dirTreeCreated
          )
      }
    } catch {
      case _: MatchError =>
        println("Image source must be one of SOHO or SDO")
        Iterable()
      case e: OutOfMemoryError =>
        e.printStackTrace()
        println("\nOut of Memory!!")
        Iterable()
      case e: Exception =>
        e.printStackTrace()
        Iterable()
    }

  def load_soho_mc(
    soho_files_path: Path,
    year_month: YearMonth,
    soho_sources: Seq[SOHO],
    dirTreeCreated: Boolean
  ): Iterable[(DateTime, (SOHO, Path))] =
    SOHOLoader.load_images(
      soho_files_path,
      year_month,
      soho_sources,
      dirTreeCreated
    )

  def load_sdo_mc(
    sdo_files_path: Path,
    year_month: YearMonth,
    sdo_sources: Seq[SDO],
    dirTreeCreated: Boolean
  ): Iterable[(DateTime, (SDO, Path))] =
    SDOLoader.load_images(
      sdo_files_path,
      year_month,
      sdo_sources,
      dirTreeCreated
    )

  def load_mc(
    files_path: Path,
    year_month: YearMonth,
    sources: Seq[SolarImagesSource],
    dirTreeCreated: Boolean
  ): Iterable[(DateTime, (SolarImagesSource, Path))] = {

    sources.flatMap(
      s =>
        s match {

          case _: SDO =>
            load_images(files_path / 'sdo, year_month, s, dirTreeCreated)
              .map(p => (p._1, (s, p._2)))

          case _: SOHO =>
            load_images(files_path / 'soho, year_month, s, dirTreeCreated)
              .map(p => (p._1, (s, p._2)))
        }
    )
  }

  /**
    * Load X-Ray fluxes averaged over all GOES missions
    *
    * */
  def load_fluxes(
    goes_files_path: Path,
    year_month: YearMonth,
    goes_source: GOES,
    dirTreeCreated: Boolean = true
  ): Stream[(DateTime, (Double, Double))] =
    GOESLoader
      .load_goes_data(goes_files_path, year_month, goes_source, dirTreeCreated)
      .map(p => {

        val data_low_wavelength  = p._2.map(_._1).filterNot(_.isNaN)
        val data_high_wavelength = p._2.map(_._2).filterNot(_.isNaN)

        val avg_low_freq = data_low_wavelength.sum / data_low_wavelength.length
        val avg_high_freq = data_high_wavelength.sum / data_high_wavelength
          .length

        (p._1, (avg_low_freq, avg_high_freq))
      })

  /**
    * Collate data from GOES with Image data.
    *
    * @param goes_data_path GOES data path.
    *
    * @param images_path path containing images.
    *
    * @param goes_aggregation The number of goes entries to group for
    *                         calculating running statistics.
    *
    * @param goes_reduce_func A function which computes some aggregation of a group
    *                         of GOES data entries.
    *
    * @param dt_round_off A function which appropriately rounds off date time instances
    *                     for the image data, enabling it to be joined to the GOES data
    *                     based on date time stamps.
    * */
  def collate_goes_data(
    year_month: YearMonth
  )(goes_source: GOES,
    goes_data_path: Path,
    goes_aggregation: Int,
    goes_reduce_func: Stream[(DateTime, (Double, Double))] => (DateTime,
      (Double, Double)),
    image_source: SOHO,
    images_path: Path,
    dt_round_off: DateTime => DateTime,
    dirTreeCreated: Boolean = true
  ): Stream[(DateTime, (Path, (Double, Double)))] = {

    val proc_goes_data =
      load_fluxes(goes_data_path, year_month, goes_source, dirTreeCreated)
        .grouped(goes_aggregation)
        .map(goes_reduce_func)
        .toMap

    val proc_image_data =
      load_images(images_path, year_month, image_source, dirTreeCreated)
        .map(p => (dt_round_off(p._1), p._2))
        .toMap

    proc_image_data
      .map(kv => {

        val value =
          if (proc_goes_data.contains(kv._1)) Some(proc_goes_data(kv._1))
          else None

        (kv._1, (kv._2, value))
      })
      .toStream
      .filter(k => k._2._2.isDefined)
      .map(k => (k._1, (k._2._1, k._2._2.get)))
      .sortBy(_._1.getMillis)
  }

  /**
    * Calls [[collate_goes_data()]] over a time period and returns the collected data.
    *
    * @param start_year_month Starting Year-Month
    * @param end_year_month Ending Year-Month
    * @param goes_data_path GOES data path.
    * @param images_path path containing images.
    * @param goes_aggregation The number of goes entries to group for
    *                         calculating running statistics.
    * @param goes_reduce_func A function which computes some aggregation of a group
    *                         of GOES data entries.
    * @param dt_round_off A function which appropriately rounds off date time instances
    *                     for the image data, enabling it to be joined to the GOES data
    *                     based on date time stamps.
    * */
  def collate_goes_data_range(
    start_year_month: YearMonth,
    end_year_month: YearMonth
  )(goes_source: GOES,
    goes_data_path: Path,
    goes_aggregation: Int,
    goes_reduce_func: Stream[(DateTime, (Double, Double))] => (DateTime,
      (Double, Double)),
    image_source: SOHO,
    images_path: Path,
    dt_round_off: DateTime => DateTime,
    dirTreeCreated: Boolean = true
  ): Stream[(DateTime, (Path, (Double, Double)))] = {

    val prepare_data = (ym: YearMonth) =>
      collate_goes_data(ym)(
        goes_source,
        goes_data_path,
        goes_aggregation,
        goes_reduce_func,
        image_source,
        images_path,
        dt_round_off
      )

    val period = new Period(
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    print("Time period considered (in months): ")

    val num_months = (12 * period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    (0 to num_months)
      .map(start_year_month.plusMonths)
      .flatMap(prepare_data)
      .toStream
  }

  /**
    * Create a joint data set between heliospheric
    * images and L1 time series.
    * @param start_year_month Starting Year-Month
    * @param end_year_month Ending Year-Month
    *
    * */
  def join_omni[T <: SolarImagesSource](
    start_year_month: YearMonth,
    end_year_month: YearMonth,
    omni_source: OMNI,
    omni_data_path: Path,
    deltaT: (Int, Int),
    image_source: T,
    images_path: Path,
    image_dir_tree: Boolean = true
  ): HELIOS_OMNI_DATA = {

    val (start_instant, end_instant) = (
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val period = new Period(start_instant, end_instant)

    print("Time period considered (in months): ")

    val num_months = (12 * period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    //Extract OMNI data as stream

    //First create the transformation pipe
    val omni_processing =
      OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(
        OMNIData.Quantities.V_SW
      ) >
        IterableDataPipe[(DateTime, Seq[Double])](
          (p: (DateTime, Seq[Double])) =>
            p._1.isAfter(start_instant) && p._1.isBefore(end_instant)
        )

    val omni_data = dtfdata
      .dataset(start_year_month.getYear to end_year_month.getYear)
      .map(
        DataPipe(
          (i: Int) =>
            omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
        )
      )
      .transform(omni_processing)
      .to_supervised(id[(DateTime, Seq[Double])])

    //Extract paths to images, along with a time-stamp

    val image_dt_roundoff = DataPipe[DateTime, DateTime](
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

    val image_processing = IterableFlatMapPipe((year_month: YearMonth) => {
      val images_for_month =
        load_images[T](images_path, year_month, image_source, image_dir_tree)

      val grouped_images = images_for_month
        .map(patt => (image_dt_roundoff(patt._1), patt._2))
        .groupBy(_._1)

      grouped_images.map(
        patt => (patt._1, patt._2.toSeq.sortBy(_._1).map(_._2))
      )
    })

    val images = dtfdata
      .dataset(0 to num_months)
      .map(DataPipe((i: Int) => start_year_month.plusMonths(i)))
      .transform(image_processing)
      .to_supervised(id[(DateTime, Seq[Path])])

    images.join(omni_data)
  }

  def join_omni(
    start_year_month: YearMonth,
    end_year_month: YearMonth,
    omni_source: OMNI,
    omni_data_path: Path,
    deltaT: (Int, Int),
    image_sources: Seq[SolarImagesSource],
    images_path: Path,
    image_dir_tree: Boolean
  ): HELIOS_MC_OMNI_DATA = {

    val (start_instant, end_instant) = (
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val period = new Period(start_instant, end_instant)

    print("Time period considered (in months): ")

    val num_months = (12 * period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    //Extract OMNI data as stream

    //First create the transformation pipe
    val omni_processing =
      OMNILoader.omniVarToSlidingTS(deltaT._1, deltaT._2)(
        OMNIData.Quantities.V_SW
      ) >
        IterableDataPipe[(DateTime, Seq[Double])](
          (p: (DateTime, Seq[Double])) =>
            p._1.isAfter(start_instant) && p._1.isBefore(end_instant)
        )

    val omni_data = dtfdata
      .dataset(start_year_month.getYear to end_year_month.getYear)
      .map(
        DataPipe(
          (i: Int) =>
            omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
        )
      )
      .transform(omni_processing)
      .to_supervised(id[(DateTime, Seq[Double])])

    //Extract paths to images, along with a time-stamp

    val image_dt_roundoff = DataPipe[DateTime, DateTime](
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

    val image_processing = IterableFlatMapPipe(
      (year_month: YearMonth) =>
        load_mc(images_path, year_month, image_sources, image_dir_tree)
    ) >
      IterableDataPipe(
        image_dt_roundoff * id[(SolarImagesSource, Path)]
      ) >
      DataPipe((d: Iterable[(DateTime, (SolarImagesSource, Path))]) => {

        val images_by_hour = d.groupBy(_._1)

        //For each hour, select the first image for each channel

        val chosen_images_by_hour
          : Iterable[(DateTime, Map[SolarImagesSource, Seq[Path]])] =
          images_by_hour.mapValues(
            _.groupBy(_._2._1)
              .map(kv => (kv._1, kv._2.toSeq.sortBy(_._1).map(_._2._2)))
          )

        chosen_images_by_hour
      })

    val images = dtfdata
      .dataset(0 to num_months)
      .map(DataPipe((i: Int) => start_year_month.plusMonths(i)))
      .transform(image_processing)
      .to_supervised(
        id[(DateTime, Map[SolarImagesSource, Seq[Path]])]
      )

    images.join(omni_data)
  }

  /**
    * Create a joint data set between heliospheric
    * images and L1 time series. Take time history of
    * omni quantity as well as future trajectory.
    *
    * @param start_year_month Starting Year-Month
    * @param end_year_month Ending Year-Month
    *
    * */
  def join_omni[T <: SolarImagesSource](
    start_year_month: YearMonth,
    end_year_month: YearMonth,
    omni_source: OMNI,
    omni_data_path: Path,
    past_history: Int,
    deltaT: (Int, Int),
    image_source: T,
    images_path: Path,
    image_dir_tree: Boolean
  ): HELIOS_OMNI_DATA_EXT = {

    val (start_instant, end_instant) = (
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val period = new Period(start_instant, end_instant)

    print("Time period considered (in months): ")

    val num_months = (12 * period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    //Extract OMNI data as stream

    //First create the transformation pipe

    val omni_processing =
      OMNILoader.omniVarToSlidingTS(past_history, deltaT._1, deltaT._2)(
        OMNIData.Quantities.V_SW
      ) >
        IterableDataPipe[(DateTime, (Seq[Double], Seq[Double]))](
          (p: (DateTime, (Seq[Double], Seq[Double]))) =>
            p._1.isAfter(start_instant) && p._1.isBefore(end_instant)
        )

    val omni_data = dtfdata
      .dataset(start_year_month.getYear to end_year_month.getYear)
      .map(
        DataPipe(
          (i: Int) =>
            omni_data_path.toString() + "/" + OMNIData.getFilePattern(i)
        )
      )
      .transform(omni_processing)
      .to_supervised(id[(DateTime, (Seq[Double], Seq[Double]))])

    //Extract paths to images, along with a time-stamp

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

    val image_processing = IterableFlatMapPipe((year_month: YearMonth) => {
      val images_for_month =
        load_images[T](images_path, year_month, image_source, image_dir_tree)

      val grouped_images = images_for_month
        .map(patt => (image_dt_roundoff(patt._1), patt._2))
        .groupBy(_._1)

      grouped_images.map(patt => (patt._1, patt._2.toSeq.minBy(_._1)._2))
    })

    val images = dtfdata
      .dataset(0 to num_months)
      .map(DataPipe((i: Int) => start_year_month.plusMonths(i)))
      .transform(image_processing)
      .to_supervised(id[(DateTime, Path)])

    images.join(omni_data)
  }

  /**
    * Create a joint data set between heliospheric
    * images and L1 time series. Take time history of
    * omni quantity as well as future trajectory.
    *
    * @param start_year_month Starting Year-Month
    * @param end_year_month Ending Year-Month
    *
    * */
  def join_omni(
    start_year_month: YearMonth,
    end_year_month: YearMonth,
    omni_source: OMNI,
    omni_data_path: Path,
    past_history: Int,
    deltaT: (Int, Int),
    image_sources: Seq[SOHO],
    images_path: Path,
    image_dir_tree: Boolean
  ): HELIOS_MC_OMNI_DATA_EXT = {

    val (start_instant, end_instant) = (
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val period = new Period(start_instant, end_instant)

    print("Time period considered (in months): ")

    val num_months = (12 * period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    //Extract OMNI data as stream

    //First create the transformation pipe

    val omni_processing =
      OMNILoader.omniVarToSlidingTS(past_history, deltaT._1, deltaT._2)(
        OMNIData.Quantities.V_SW
      ) >
        DataPipe[Iterable[(DateTime, (Seq[Double], Seq[Double]))], Stream[
          (DateTime, (Seq[Double], Seq[Double]))
        ]](_.toStream) >
        StreamDataPipe[(DateTime, (Seq[Double], Seq[Double]))](
          (p: (DateTime, (Seq[Double], Seq[Double]))) =>
            p._1.isAfter(start_instant) && p._1.isBefore(end_instant)
        )

    val years = (start_year_month.getYear to end_year_month.getYear).toStream

    val omni_data = omni_processing(
      years
        .map(i => omni_data_path.toString() + "/" + OMNIData.getFilePattern(i))
    )

    //Extract paths to images, along with a time-stamp

    val image_dt_roundoff: DataPipe[DateTime, DateTime] = DataPipe(
      (d: DateTime) => {
        new DateTime(
          d.getYear,
          d.getMonthOfYear,
          d.getDayOfMonth,
          d.getHourOfDay,
          0,
          0
        )
      }
    )

    val image_processing = StreamFlatMapPipe(
      (year_month: YearMonth) =>
        load_soho_mc(images_path, year_month, image_sources, image_dir_tree).toStream
    ) >
      StreamDataPipe(image_dt_roundoff * id[(SOHO, Path)]) >
      DataPipe(
        (d: Stream[(DateTime, (SOHO, Path))]) =>
          d.groupBy(_._1)
            .mapValues(_.map(_._2).groupBy(_._1).mapValues(_.map(_._2).toSeq))
      )

    val images = image_processing(
      (0 to num_months).map(start_year_month.plusMonths).toStream
    )

    omni_data
      .map(o => {
        val image_option = images.get(o._1)
        (o._1, image_option, o._2)
      })
      .filter(_._2.isDefined)
      .map(d => (d._1, (d._2.get, d._3)))

  }

  /**
    * Resample data according to a provided
    * bounded discrete random variable
    * */
  def resample[T](
    data: Stream[T],
    selector: DiscreteDistrRV[Int]
  ): Stream[T] = {

    //Resample training set ot
    //emphasize extreme events.
    println("\nResampling data instances\n")

    selector.iid(data.length).draw.map(data(_))
  }

  def resample_op[T](resample_flag: Boolean) =
    DataPipe(
      (d: Iterable[(T, Seq[Double])]) =>
        if (resample_flag) {

          /*
           * Resample training set with
           * emphasis on larger ratios
           * between max and min of a sliding
           * time window.
           * */
          val un_prob: Array[Double] = d
            .map(p => {

              math.abs(p._2.max - p._2.min) / math.abs(p._2.min)
            })
            .map(math.exp)
            .toArray

          val normalizer = un_prob.sum

          val selector = MultinomialRV(DenseVector(un_prob) / normalizer)

          helios.data.resample[(T, Seq[Double])](d.toStream, selector)

        } else d
    )

  def resample_op_ts[T](resample_flag: Boolean) =
    DataPipe(
      (d: Iterable[(DateTime, (T, Seq[Double]))]) =>
        if (resample_flag) {

          /*
           * Resample training set with
           * emphasis on larger ratios
           * between max and min of a sliding
           * time window.
           * */
          val un_prob: Array[Double] = d
            .map(p => {

              math.abs(p._2._2.max - p._2._2.min) / math.abs(p._2._2.min)
            })
            .map(math.exp)
            .toArray

          val normalizer = un_prob.sum

          val selector = MultinomialRV(DenseVector(un_prob) / normalizer)

          helios.data
            .resample[(DateTime, (T, Seq[Double]))](d.toStream, selector)

        } else d
    )

  private def print_data_splits(train_fraction: Double): Unit = {
    print("Training: % ")
    pprint.pprintln(train_fraction)

    print("Test:     % ")
    val test_fraction = math.round(100 * (100f - train_fraction.toFloat)) / 100d
    pprint.pprintln(test_fraction)
  }

  def create_double_tensor_buffered(
    buff_size: Int
  )(coll: Iterable[Seq[Double]],
    size: Int
  ): Tensor[Double] = {

    val dimensions = coll.head.length

    println()
    val tensor_splits = coll
      .grouped(buff_size)
      .toIterable
      .zipWithIndex
      .map(splitAndIndex => {

        val split_seq = splitAndIndex._1.toSeq

        val progress = splitAndIndex._2 * buff_size * 100.0 / size

        print("Progress %:\t")
        pprint.pprintln(progress)

        dtf.tensor_from[Double](split_seq.length, dimensions)(
          split_seq.flatten[Double]
        )

      })

    dtf.concatenate(tensor_splits.toSeq, axis = 0)
  }

  def prepare_helios_data_set(
    collated_data: HELIOS_IMAGE_DATA,
    read_image: DataPipe[Path, Tensor[UByte]],
    tt_partition: IMAGE_PATTERN => Boolean,
    image_history: Int,
    image_history_downsampling: Int
  ): TF_IMAGE_DATA[UByte] = {

    println("Separating data into train and test.\n")
    val experiment_data = collated_data.partition(DataPipe(tt_partition))

    print("Total data size: ")
    val total_data_size = collated_data.size

    pprint.pprintln(total_data_size)

    val train_data_size = experiment_data.training_dataset.size

    val train_fraction = math.round(
      100 * train_data_size.toFloat * 100 / total_data_size
    ) / 100d

    print_data_splits(train_fraction)

    val trim_time_stamp = DataPipe((p: IMAGE_PATTERN) => p._2)

    val load_only_images = trim_time_stamp > read_image

    val get_image_history = if (image_history > 0) {
      val slices = utils
        .range(min = 0d, image_history.toDouble, image_history_downsampling)
        .map(_.toInt)

      val indices = slices :+ (image_history - 1)

      DataPipe(
        (data_stream: Iterable[Tensor[UByte]]) =>
          data_stream
            .sliding(image_history)
            .map(group => {

              val gr = group.toSeq

              val images = tfi.concatenate(indices.map(gr(_)), axis = -1)

              images
            })
            .toIterable
      )

    } else {
      id[Iterable[Tensor[UByte]]]
    }

    experiment_data.copy[Tensor[UByte]](
      training_dataset = experiment_data.training_dataset
        .map(load_only_images)
        .transform(get_image_history),
      test_dataset = experiment_data.test_dataset
        .map(load_only_images)
        .transform(get_image_history)
    )
  }

  def write_helios_data_set(
    dataset: HELIOS_OMNI_DATA,
    directory: Path,
    identifier: String
  ): Unit = {

    val pattern_to_map = DataPipe[PATTERN, JValue](
      p =>
        (
          ("timestamp" -> p._1.toString("yyyy-MM-dd'T'HH:mm:ss'Z'")) ~
            ("images"  -> p._2._1.map(x => x.toString)) ~
            ("targets" -> p._2._2)
        )
    )

    val map_to_json = DataPipe[JValue, String](p => write_json(p))

    val process_pattern = pattern_to_map > map_to_json

    val json_records = dataset.map(process_pattern).data.mkString(",\n")

    write.over(directory / s"$identifier.json", s"$json_records")
  }

  /**
    * Create a processed tensor data set as a [[TF_DATA]] instance.
    *
    * @param collated_data A Stream of date times, image paths and outputs.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * */
  def prepare_helios_data_set[T: TF: IsFloatOrDouble](
    collated_data: HELIOS_OMNI_DATA,
    read_image: DataPipe[Seq[Path], Option[Tensor[UByte]]],
    read_targets: DataPipe[Seq[Double], Tensor[T]],
    tt_partition: PATTERN => Boolean,
    resample: Boolean = false,
    image_history: Int = 0,
    image_history_downsampling: Int = 0
  ): TF_DATA[UByte, T] = {

    println("Separating data into train and test.\n")
    val experiment_data = collated_data.partition(DataPipe(tt_partition))

    print("Total data size: ")
    val total_data_size = collated_data.size

    pprint.pprintln(total_data_size)

    val train_data_size = experiment_data.training_dataset.size

    val train_fraction = math.round(
      100 * train_data_size.toFloat * 100 / total_data_size
    ) / 100d

    print_data_splits(train_fraction)

    val trim_time_stamp = DataPipe((p: PATTERN) => p._2)

    val load_only_images = trim_time_stamp > (read_image * id[Seq[
      Double
    ]])

    val load_only_targets = id[Tensor[UByte]] * read_targets

    val get_image_history = if (image_history > 0) {
      val slices = utils
        .range(min = 0d, image_history.toDouble, image_history_downsampling)
        .map(_.toInt)

      val indices = slices :+ (image_history - 1)

      DataPipe(
        (data_stream: Iterable[(Tensor[UByte], Seq[Double])]) =>
          data_stream
            .sliding(image_history)
            .map(group => {

              val gr = group.toSeq

              val images =
                tfi.concatenate(indices.map(i => gr(i)._1), axis = -1)

              (images, gr(indices.last)._2)
            })
            .toIterable
      )

    } else {
      id[Iterable[(Tensor[UByte], Seq[Double])]]
    }

    val processed_data = experiment_data.copy[(Tensor[UByte], Seq[Double])](
      training_dataset = experiment_data.training_dataset
        .map(load_only_images)
        .filter(non_corrupted_pattern)
        .map(
          get_image_tensor
            * id[Seq[Double]]
        )
        .transform(get_image_history),
      test_dataset = experiment_data.test_dataset
        .map(load_only_images)
        .filter(non_corrupted_pattern)
        .map(
          get_image_tensor
            * id[Seq[Double]]
        )
        .transform(get_image_history)
    )

    processed_data.copy[(Tensor[UByte], Tensor[T])](
      training_dataset = processed_data.training_dataset
        .transform(resample_op[Tensor[UByte]](resample))
        .map(load_only_targets),
      test_dataset = processed_data.test_dataset
        .map(load_only_targets)
    )
  }

  /**
    * Create a processed tensor data set as a [[TF_DATA]] instance.
    *
    * @param collated_data A Stream of date times, image paths and outputs.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * */
  def time_stamped_data_set[T: TF: IsFloatOrDouble](
    collated_data: HELIOS_OMNI_DATA,
    read_image: DataPipe[Seq[Path], Option[Tensor[UByte]]],
    read_targets: DataPipe[Seq[Double], Tensor[T]],
    tt_partition: PATTERN => Boolean,
    resample: Boolean = false,
    image_history: Int = 0,
    image_history_downsampling: Int = 0
  ): TF_DATA_T[UByte, T] = {

    println("Separating data into train and test.\n")
    val experiment_data = collated_data.partition(DataPipe(tt_partition))

    print("Total data size: ")
    val total_data_size = collated_data.size

    pprint.pprintln(total_data_size)

    val train_data_size = experiment_data.training_dataset.size

    val train_fraction = math.round(
      100 * train_data_size.toFloat * 100 / total_data_size
    ) / 100d

    print_data_splits(train_fraction)

    val trim_time_stamp = DataPipe((p: PATTERN) => p._2)

    val load_only_images =
      id[DateTime] * (read_image * id[Seq[Double]])

    val load_only_targets = id[DateTime] * (id[Tensor[UByte]] * read_targets)

    val get_image_history = if (image_history > 0) {
      val slices = utils
        .range(min = 0d, image_history.toDouble, image_history_downsampling)
        .map(_.toInt)

      val indices = slices :+ (image_history - 1)

      DataPipe(
        (data_stream: Iterable[(DateTime, (Tensor[UByte], Seq[Double]))]) =>
          data_stream
            .sliding(image_history)
            .map(group => {

              val gr = group.toSeq

              val images =
                tfi.concatenate(indices.map(i => gr(i)._2._1), axis = -1)

              (gr(indices.last)._1, (images, gr(indices.last)._2._2))
            })
            .toIterable
      )

    } else {
      id[Iterable[(DateTime, (Tensor[UByte], Seq[Double]))]]
    }

    val non_corrupted_images =
      DataPipe[
        (DateTime, (Option[Tensor[UByte]], Seq[Double])),
        (Option[Tensor[UByte]], Seq[Double])
      ](_._2) >
        non_corrupted_pattern

    val processed_data =
      experiment_data.copy[(DateTime, (Tensor[UByte], Seq[Double]))](
        training_dataset = experiment_data.training_dataset
          .map(load_only_images)
          .filter(non_corrupted_images)
          .map(id[DateTime] * (get_image_tensor * id[Seq[Double]]))
          .transform(get_image_history),
        test_dataset = experiment_data.test_dataset
          .map(load_only_images)
          .filter(non_corrupted_images)
          .map(id[DateTime] * (get_image_tensor * id[Seq[Double]]))
          .transform(get_image_history)
      )

    processed_data.copy[(DateTime, (Tensor[UByte], Tensor[T]))](
      training_dataset = processed_data.training_dataset
        .transform(resample_op_ts[Tensor[UByte]](resample))
        .map(load_only_targets),
      test_dataset = processed_data.test_dataset
        .map(load_only_targets)
    )
  }

  /**
    * Create a processed tensor data set as a [[AbstractDataSet]] instance.
    *
    * @param collated_data A Stream of date times, image paths output histories and outputs.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * */
  def prepare_helios_ts_data_set[T: TF: IsFloatOrDouble](
    collated_data: HELIOS_OMNI_DATA_EXT,
    read_image: DataPipe[Path, Tensor[UByte]],
    read_targets: DataPipe[Seq[Double], Tensor[T]],
    read_targets_history: DataPipe[Seq[Double], Tensor[T]],
    tt_partition: PATTERN_EXT => Boolean,
    resample: Boolean = false,
    image_history: Int = 0,
    image_history_downsampling: Int = 0
  ): TF_DATA_EXT[UByte, T] = {

    println("Separating data into train and test.\n")
    val experiment_data = collated_data.partition(DataPipe(tt_partition))

    print("Total data size: ")
    val total_data_size = collated_data.size

    pprint.pprintln(total_data_size)

    val train_data_size = experiment_data.training_dataset.size

    val train_fraction = math.round(
      100 * train_data_size.toFloat * 100 / total_data_size
    ) / 100d

    print_data_splits(train_fraction)

    val trim_time_stamp = DataPipe((p: PATTERN_EXT) => p._2)

    val load_only_images = trim_time_stamp > (read_image * id[
      (Seq[Double], Seq[Double])
    ])

    type Patt = (Tensor[UByte], (Seq[Double], Seq[Double]))

    val load_hist_and_targets
      : DataPipe[Patt, (Tensor[UByte], (Tensor[T], Tensor[T]))] =
      id[Tensor[UByte]] * (read_targets_history * read_targets)

    val get_image_history = if (image_history > 0) {
      val slices = utils
        .range(min = 0d, image_history.toDouble, image_history_downsampling)
        .map(_.toInt)

      val indices = slices :+ (image_history - 1)

      DataPipe(
        (data_stream: Iterable[(Tensor[UByte], (Seq[Double], Seq[Double]))]) =>
          data_stream
            .sliding(image_history)
            .map(group => {

              val gr = group.toSeq

              val images =
                tfi.concatenate(indices.map(i => gr(i)._1), axis = -1)

              (images, gr(indices.last)._2)
            })
            .toIterable
      )

    } else {
      id[Iterable[(Tensor[UByte], (Seq[Double], Seq[Double]))]]
    }

    val processed_data =
      experiment_data.copy[(Tensor[UByte], (Seq[Double], Seq[Double]))](
        training_dataset = experiment_data.training_dataset
          .map(load_only_images)
          .transform(get_image_history),
        test_dataset = experiment_data.test_dataset
          .map(load_only_images)
          .transform(get_image_history)
      )

    /*
     * If the `resample` flag is set to true,
     * balance the occurrences of high and low
     * flux events through re-sampling.
     *
     * */

    type Coll = Iterable[Patt]

    val resample_op: DataPipe[Coll, Coll] = DataPipe[Coll, Coll](
      (d: Coll) =>
        if (resample) {

          /*
           * Resample training set with
           * emphasis on larger ratios
           * between max and min of a sliding
           * time window.
           * */
          val un_prob: Array[Double] = d
            .map(p => {

              math.abs(p._2._2.max - p._2._2.min) / math.abs(p._2._2.min)
            })
            .map(math.exp)
            .toArray

          val normalizer = un_prob.sum

          val selector = MultinomialRV(DenseVector(un_prob) / normalizer)

          helios.data.resample[Patt](d.toStream, selector)

        } else d
    )

    val pattern_rearrange = DataPipe(
      (pattern: (Tensor[UByte], (Tensor[T], Tensor[T]))) =>
        ((pattern._1, pattern._2._1), pattern._2._2)
    )

    processed_data.copy(
      training_dataset = processed_data.training_dataset
        .transform[Patt](resample_op)
        .map(load_hist_and_targets)
        .map(pattern_rearrange),
      test_dataset = processed_data.test_dataset
        .map(load_hist_and_targets)
        .map(pattern_rearrange)
    )
  }

  /**
    * Create a processed tensor data set as a [[TF_DATA]] instance.
    *
    * @param collated_data A Stream of date times, image paths and outputs.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * */
  def prepare_mc_helios_data_set[T: TF: IsFloatOrDouble](
    image_sources: Seq[SolarImagesSource],
    collated_data: HELIOS_MC_OMNI_DATA,
    read_mc_image: DataPipe[Map[SolarImagesSource, Seq[Path]], Option[
      Tensor[UByte]
    ]],
    read_targets: DataPipe[Seq[Double], Tensor[T]],
    tt_partition: MC_PATTERN => Boolean,
    resample: Boolean = false,
    image_history: Int = 0,
    image_history_downsampling: Int = 0
  ): TF_DATA[UByte, T] = {

    println("Separating data into train and test.\n")
    val experiment_data = collated_data.partition(DataPipe(tt_partition))

    print("Total data size: ")
    val total_data_size = collated_data.size

    pprint.pprintln(total_data_size)

    val train_data_size = experiment_data.training_dataset.size

    val train_fraction = math.round(
      100 * train_data_size.toFloat * 100 / total_data_size
    ) / 100d

    print_data_splits(train_fraction)

    val trim_time_stamp = DataPipe((p: MC_PATTERN) => p._2)

    val load_only_images = trim_time_stamp > (read_mc_image * id[Seq[
      Double
    ]])

    val load_only_targets = id[Tensor[UByte]] * read_targets

    val get_image_history = if (image_history > 0) {
      val slices = utils
        .range(min = 0d, image_history.toDouble, image_history_downsampling)
        .map(_.toInt)

      val indices = slices :+ (image_history - 1)

      DataPipe(
        (data_stream: Iterable[(Tensor[UByte], Seq[Double])]) =>
          data_stream
            .sliding(image_history)
            .map(group => {
              val gr = group.toSeq

              val images: Tensor[UByte] =
                tfi.concatenate(indices.map(i => gr(i)._1), axis = -1)

              (images, gr(indices.last)._2)
            })
            .toIterable
      )

    } else {
      id[Iterable[(Tensor[UByte], Seq[Double])]]
    }

    experiment_data.copy[(Tensor[UByte], Tensor[T])](
      training_dataset = experiment_data.training_dataset
        .map(load_only_images)
        .filter(non_corrupted_pattern)
        .map(
          get_image_tensor
            * id[Seq[Double]]
        )
        .transform(get_image_history > resample_op[Tensor[UByte]](resample))
        .map(load_only_targets),
      test_dataset = experiment_data.test_dataset
        .map(load_only_images)
        .filter(non_corrupted_pattern)
        .map(
          get_image_tensor
            * id[Seq[Double]]
        )
        .transform(get_image_history > resample_op[Tensor[UByte]](resample))
        .map(load_only_targets)
    )
  }

  def prepare_mc_helios_ts_data_set[T: TF: IsFloatOrDouble](
    image_sources: Seq[SOHO],
    collated_data: HELIOS_MC_OMNI_DATA_EXT,
    tt_partition: MC_PATTERN_EXT => Boolean,
    image_process: Map[SOHO, DataPipe[Image, Image]],
    images_to_bytes: DataPipe[Seq[Image], Array[Byte]],
    resample: Boolean
  ): TF_MC_DATA_EXT[UByte, T] = {

    println()
    println("Filtering complete data patterns")
    val complete_data = collated_data.filter(
      p =>
        image_sources.forall(s => p._2._1.keys.toSeq.contains(s)) &&
          p._2._1.values.forall(s => s.nonEmpty)
    )

    print("Total data size: ")
    pprint.pprintln(collated_data.toIterator.length)

    print("Usable data size: ")
    pprint.pprintln(complete_data.toIterator.length)
    println()

    println("Separating data into train and test.\n")
    val (train_set, test_set) = complete_data.partition(tt_partition)

    print("Total data size: ")
    val total_data_size = complete_data.toIterator.length

    pprint.pprintln(total_data_size)

    val train_data_size = train_set.toIterator.length
    val test_data_size  = test_set.toIterator.length

    val train_fraction = math.round(
      100 * train_data_size.toFloat * 100 / total_data_size
    ) / 100d

    print_data_splits(train_fraction)

    //Calculate the height, width and number of channels
    //in the images
    val (scaled_height, scaled_width, num_channels) = {

      val scaled_image = image_process(train_set.head._2._1.keys.head)(
        Image.fromPath(train_set.head._2._1.values.head.head.toNIO)
      )

      (scaled_image.height, scaled_image.width, image_sources.length)

    }

    val working_set =
      TF_DATA_EXT(null, null, train_data_size, null, null, test_data_size)

    /*
     * If the `resample` flag is set to true,
     * balance the occurence of high and low
     * flux events through re-sampling.
     *
     * */
    val processed_train_set = if (resample) {

      /*
       * Resample training set with
       * emphasis on larger ratios
       * between max and min of a sliding
       * time window.
       * */
      val un_prob = train_set
        .map(p => {

          math.abs(p._2._2._2.max - p._2._2._2.min) / math.abs(p._2._2._2.min)
        })
        .map(math.exp)

      val normalizer = un_prob.sum

      val selector = MultinomialRV(DenseVector(un_prob.toArray) / normalizer)

      helios.data.resample(train_set.toStream, selector)
    } else train_set

    def split_features_and_labels(
      coll: HELIOS_MC_OMNI_DATA_EXT
    ): (Iterable[(Map[SOHO, Seq[Path]], Seq[Double])], Iterable[Seq[Double]]) =
      coll
        .map(entry => {

          val (_, (images_map, (data_history, data_label))) = entry

          ((images_map, data_history), data_label)

        })
        .unzip

    println()
    //Construct training features and labels
    println("Processing Training Data Set")
    val (features_train, labels_train) = split_features_and_labels(
      processed_train_set.toStream
    )

    println("Loading \n\t1) image features \n\t2) time series history")
    val features_tensor_train = (
      dtfdata.create_image_tensor_buffered(
        size_buffer,
        image_sources,
        image_process,
        images_to_bytes,
        scaled_height,
        scaled_width,
        num_channels
      )(features_train.map(_._1), train_data_size),
      create_double_tensor_buffered(size_buffer)(
        features_train.map(_._2),
        train_data_size
      ).castTo[T]
    )

    println("Loading targets")
    val labels_tensor_train =
      create_double_tensor_buffered(size_buffer)(labels_train, train_data_size)
        .castTo[T]

    println()
    //Construct test features and labels
    println("Processing Test Data Set")
    val (features_test, labels_test) = split_features_and_labels(
      test_set.toStream
    )

    println("Loading \n\t1) image features \n\t2) time series history")
    val features_tensor_test = (
      dtfdata.create_image_tensor_buffered(
        size_buffer,
        image_sources,
        image_process,
        images_to_bytes,
        scaled_height,
        scaled_width,
        num_channels
      )(features_test.map(_._1), test_data_size),
      create_double_tensor_buffered(size_buffer)(
        features_test.map(_._2),
        test_data_size
      ).castTo[T]
    )

    println("Loading targets")
    val labels_tensor_test =
      create_double_tensor_buffered(size_buffer)(labels_test, test_data_size)
        .castTo[T]

    println("Helios data set created\n")
    working_set.copy(
      trainData = features_tensor_train,
      trainLabels = labels_tensor_train,
      testData = features_tensor_test,
      testLabels = labels_tensor_test
    )
  }

  /**
    * Create a processed tensor data set as a [[HeliosDataSet]] instance.
    *
    * @param collated_data A Stream of date times, image paths and fluxes.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * @param scaleDownFactor The exponent of 2 which determines how much the
    *                        image will be scaled down. i.e. scaleDownFactor = 4
    *                        corresponds to a 16 fold decrease in image size.
    * */
  def prepare_helios_goes_data_set[T: TF: IsFloatOrDouble](
    collated_data: Stream[(DateTime, (Path, (Double, Double)))],
    tt_partition: ((DateTime, (Path, (Double, Double)))) => Boolean,
    scaleDownFactor: Int = 4,
    resample: Boolean = false
  ): HeliosDataSet[UByte, T] = {

    val scaleDown = 1 / math.pow(2, scaleDownFactor)

    print("Scaling down images by a factor of ")
    pprint.pprintln(math.pow(2, scaleDownFactor))
    println()

    println("Separating data into train and test.\n")
    val (train_set, test_set) = collated_data.partition(tt_partition)

    //Calculate the height, width and number of channels
    //in the images
    val (scaled_height, scaled_width, num_channels) = {

      val im = Image.fromPath(train_set.head._2._1.toNIO)

      val scaled_image = im.copy.scale(scaleDown)

      (scaled_image.height, scaled_image.width, scaled_image.argb(0, 0).length)

    }

    val working_set =
      HeliosDataSet(null, null, train_set.length, null, null, test_set.length)

    /*
     * If the `resample` flag is set to true,
     * balance the occurence of high and low
     * flux events through re-sampling.
     *
     * */
    val processed_train_set = if (resample) {
      //Resample training set ot
      //emphasize extreme events.

      val un_prob    = train_set.map(_._2._2._1).map(math.exp)
      val normalizer = un_prob.sum
      val selector   = MultinomialRV(DenseVector(un_prob.toArray) / normalizer)

      helios.data.resample(train_set, selector)
    } else train_set

    val (features_train, labels_train): (
      Stream[Array[Byte]],
      Stream[Seq[Double]]
    ) =
      processed_train_set
        .map(entry => {
          val (_, (path, data_label)) = entry

          val im = Image.fromPath(path.toNIO)

          val scaled_image = im.copy.scale(scaleDown)

          (
            scaled_image.argb.flatten.map(_.toByte),
            Seq(data_label._1, data_label._2)
          )

        })
        .unzip

    val features_tensor_train = dtf.tensor_from_buffer[UByte](
      processed_train_set.length,
      scaled_height,
      scaled_width,
      num_channels
    )(features_train.toArray.flatten[Byte])

    val labels_tensor_train = dtf
      .tensor_from[Double](train_set.length, 2)(labels_train.flatten[Double])
      .castTo[T]

    val (features_test, labels_test): (Stream[Array[Byte]], Stream[Seq[Double]]) =
      test_set
        .map(entry => {
          val (_, (path, data_label)) = entry

          val im = Image.fromPath(path.toNIO)

          val scaled_image = im.copy.scale(scaleDown)

          (
            scaled_image.argb.flatten.map(_.toByte),
            Seq(data_label._1, data_label._2)
          )

        })
        .unzip

    val features_tensor_test = dtf.tensor_from_buffer[UByte](
      test_set.length,
      scaled_height,
      scaled_width,
      num_channels
    )(features_test.toArray.flatten[Byte])

    val labels_tensor_test = dtf
      .tensor_from[Double](test_set.length, 2)(labels_test.flatten[Double])
      .castTo[T]

    println("Helios data set created\n")
    working_set.copy(
      trainData = features_tensor_train,
      trainLabels = labels_tensor_train,
      testData = features_tensor_test,
      testLabels = labels_tensor_test
    )
  }

  /**
    * Generate a starting data set for GOES prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]] method.
    *
    * @param image_source The [[SOHO]] data source to extract from
    * @param year_start The starting time of the data
    * @param year_end The end time of the data.
    * */
  def generate_data_goes(
    year_start: Int = 2001,
    year_end: Int = 2005,
    image_source: SOHO = SOHO(SOHOData.Instruments.MDIMAG, 512)
  ): Stream[(DateTime, (Path, (Double, Double)))] = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: " + os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: " + user_name)

    val home_dir_prefix =
      if (os_name.startsWith("Mac")) root / "Users" else root / 'home

    require(year_end > year_start, "Data set must encompass more than one year")

    /*
     * Create a collated data set,
     * extract GOES flux data and join it
     * with eit195 (green filter) images.
     * */

    print("Looking for data in directory ")
    val data_dir = home_dir_prefix / user_name / "data_repo" / 'helios
    pprint.pprintln(data_dir)

    val soho_dir = data_dir / 'soho
    val goes_dir = data_dir / 'goes

    val reduce_fn = (gr: Stream[(DateTime, (Double, Double))]) => {

      val (max_flux_short, max_flux_long) =
        (gr.map(_._2._1).max, gr.map(_._2._2).max)

      (gr.head._1, (math.log10(max_flux_short), math.log10(max_flux_long)))
    }

    val round_date = (d: DateTime) => {

      val num_minutes = 5

      val minutes: Int = d.getMinuteOfHour / num_minutes

      new DateTime(
        d.getYear,
        d.getMonthOfYear,
        d.getDayOfMonth,
        d.getHourOfDay,
        minutes * num_minutes
      )
    }

    println("Preparing data-set as a Stream ")
    println("Start: " + year_start + " End: " + year_end)

    collate_goes_data_range(
      new YearMonth(year_start, 1),
      new YearMonth(year_end, 12)
    )(
      GOES(GOESData.Quantities.XRAY_FLUX_5m),
      goes_dir,
      goes_aggregation = 1,
      goes_reduce_func = reduce_fn,
      image_source,
      soho_dir,
      dt_round_off = round_date
    )

  }

  /**
    * Generate a starting data set for unsupervised tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]]
    * or [[data.SDOLoader.bulk_download()]] methods.
    *
    * @param image_source The image data source to extract from
    * @param year_range The range of years, for constructing the data,
    *                   ex: (2000 to 2002)
    * */
  def generate_image_data[T <: SolarImagesSource](
    year_range: Range,
    image_source: T = SOHO(SOHOData.Instruments.MDIMAG, 512),
    images_data_dir: Option[Path] = None
  ): HELIOS_IMAGE_DATA = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: " + os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: " + user_name)

    val home_dir_prefix =
      if (os_name.startsWith("Mac")) root / "Users" else root / 'home

    print("Looking for data in directory ")
    val data_dir = images_data_dir match {
      case None       => home_dir_prefix / user_name / "data_repo" / 'helios
      case Some(path) => path
    }

    pprint.pprintln(data_dir)

    val images_dir = image_source match {
      case _: SOHO => data_dir / 'soho
      case _: SDO  => data_dir / 'sdo
      case _       => data_dir
    }

    println("Preparing data-set as a Stream ")
    print("Start: ")
    pprint.pprintln(year_range.min)
    print("End: ")
    pprint.pprintln(year_range.max)
    println()

    val (start_year_month, end_year_month) = (
      new YearMonth(year_range.min, 1),
      new YearMonth(year_range.max, 12)
    )

    val (start_instant, end_instant) = (
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val period = new Period(start_instant, end_instant)

    print("Time period considered (in months): ")

    val num_months = (12 * period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    //Extract paths to images, along with a time-stamp

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

    val image_processing = IterableFlatMapPipe((year_month: YearMonth) => {
      val images_for_month =
        load_images[T](images_dir, year_month, image_source, true)

      val grouped_images = images_for_month
        .map(patt => (image_dt_roundoff(patt._1), patt._2))
        .groupBy(_._1)

      grouped_images.map(patt => (patt._1, patt._2.toSeq.minBy(_._1)._2))
    })

    dtfdata
      .dataset(0 to num_months)
      .map(DataPipe((i: Int) => start_year_month.plusMonths(i)))
      .transform(image_processing)
  }

  /**
    * Generate a starting data set for OMNI/L1 prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]]
    * or [[data.SDOLoader.bulk_download()]] methods.
    *
    * @param image_source The image data source to extract from
    * @param year_range The range of years, for constructing the data,
    *                   ex: (2000 to 2002)
    * */
  def generate_data_omni[T <: SolarImagesSource](
    year_range: Range,
    image_source: T = SOHO(SOHOData.Instruments.MDIMAG, 512),
    omni_source: OMNI = OMNI(OMNIData.Quantities.V_SW),
    deltaT: (Int, Int) = (18, 56),
    images_data_dir: Option[Path] = None
  ): HELIOS_OMNI_DATA = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: " + os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: " + user_name)

    val home_dir_prefix =
      if (os_name.startsWith("Mac")) root / "Users" else root / 'home

    print("Looking for data in directory ")
    val data_dir = images_data_dir match {
      case None       => home_dir_prefix / user_name / "data_repo" / 'helios
      case Some(path) => path
    }

    pprint.pprintln(data_dir)

    val images_dir = image_source match {
      case _: SOHO => data_dir / 'soho
      case _: SDO  => data_dir / 'sdo
      case _       => data_dir
    }

    println("Preparing data-set as a Stream ")
    print("Start: ")
    pprint.pprintln(year_range.min)
    print("End: ")
    pprint.pprintln(year_range.max)
    println()

    join_omni[T](
      new YearMonth(year_range.min, 1),
      new YearMonth(year_range.max, 12),
      omni_source,
      pwd / "data",
      deltaT,
      image_source,
      images_dir
    )
  }

  /**
    * Generate a starting data set for OMNI/L1 prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]]
    * or [[data.SDOLoader.bulk_download()]] methods.
    *
    * @param image_sources The image data sources to extract from
    * @param year_range The range of years, for constructing the data,
    *                   ex: (2000 to 2002)
    * */
  def generate_data_mc_omni(
    year_range: Range,
    image_sources: Seq[SolarImagesSource],
    omni_source: OMNI = OMNI(OMNIData.Quantities.V_SW),
    deltaT: (Int, Int) = (18, 56),
    images_data_dir: Option[Path] = None
  ): HELIOS_MC_OMNI_DATA = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: " + os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: " + user_name)

    val home_dir_prefix =
      if (os_name.startsWith("Mac")) root / "Users" else root / 'home

    print("Looking for data in directory ")
    val data_dir = images_data_dir match {
      case None       => home_dir_prefix / user_name / "data_repo" / 'helios
      case Some(path) => path
    }

    pprint.pprintln(data_dir)

    println("Preparing data-set as a Stream ")
    print("Start: ")
    pprint.pprintln(year_range.min)
    print("End: ")
    pprint.pprintln(year_range.max)
    println()

    join_omni(
      new YearMonth(year_range.min, 1),
      new YearMonth(year_range.max, 12),
      omni_source,
      pwd / "data",
      deltaT,
      image_sources,
      data_dir,
      image_dir_tree = true
    )
  }

  /**
    * Generate a starting data set for OMNI/L1 prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]]
    * or [[data.SDOLoader.bulk_download()]] methods.
    *
    * @param image_source The image data source to extract from
    * @param year_range   The range of years, for constructing the data,
    *                     ex: (2000 to 2002)
    * */
  def generate_data_omni_ext[T <: SolarImagesSource](
    year_range: Range,
    image_source: T = SOHO(SOHOData.Instruments.MDIMAG, 512),
    omni_source: OMNI = OMNI(OMNIData.Quantities.V_SW),
    history: Int = 8,
    deltaT: (Int, Int) = (18, 56),
    images_data_dir: Option[Path] = None
  ): HELIOS_OMNI_DATA_EXT = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: " + os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: " + user_name)

    val home_dir_prefix =
      if (os_name.startsWith("Mac")) root / "Users" else root / 'home

    //require(year_end > year_start, "Data set must encompass more than one year")

    print("Looking for data in directory ")

    val data_dir = images_data_dir match {
      case None       => home_dir_prefix / user_name / "data_repo" / 'helios
      case Some(path) => path
    }

    pprint.pprintln(data_dir)

    val images_dir = image_source match {
      case _: SOHO => data_dir / 'soho
      case _: SDO  => data_dir / 'sdo
      case _       => data_dir
    }

    println("Preparing data-set as a Stream ")
    print("Start: ")
    pprint.pprintln(year_range.min)
    print("End: ")
    pprint.pprintln(year_range.max)
    println()

    join_omni[T](
      new YearMonth(year_range.min, 1),
      new YearMonth(year_range.max, 12),
      omni_source,
      pwd / "data",
      history,
      deltaT,
      image_source,
      images_dir,
      image_dir_tree = true
    )
  }

  /**
    * Generate a starting data set for OMNI/L1 prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]] method.
    *
    * @param image_sources A sequence of [[SOHO]] data source to extract from.
    * @param year_range The range of years, for constructing the data,
    *                   ex: (2000 to 2002)
    * */
  def generate_data_mc_omni_ext(
    year_range: Range,
    image_sources: Seq[SOHO] = Seq(SOHO(SOHOData.Instruments.MDIMAG, 512)),
    omni_source: OMNI = OMNI(OMNIData.Quantities.V_SW),
    history: Int = 8,
    deltaT: (Int, Int) = (18, 56)
  ): HELIOS_MC_OMNI_DATA_EXT = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: " + os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: " + user_name)

    val home_dir_prefix =
      if (os_name.startsWith("Mac")) root / "Users" else root / 'home

    //require(year_end > year_start, "Data set must encompass more than one year")

    print("Looking for data in directory ")
    val data_dir = home_dir_prefix / user_name / "data_repo" / 'helios
    pprint.pprintln(data_dir)

    val soho_dir = data_dir / 'soho

    println("Preparing data-set as a Stream ")
    print("Start: ")
    pprint.pprintln(year_range.min)
    print("End: ")
    pprint.pprintln(year_range.max)
    println()

    join_omni(
      new YearMonth(year_range.min, 1),
      new YearMonth(year_range.max, 12),
      omni_source,
      pwd / "data",
      history,
      deltaT,
      image_sources,
      soho_dir,
      image_dir_tree = true
    )
  }

}
