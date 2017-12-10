package io.github.mandar2812.PlasmaML

import ammonite.ops.Path
import com.sksamuel.scrimage.Image
import io.github.mandar2812.PlasmaML.helios.data._
import io.github.mandar2812.dynaml.tensorflow.dtf
import org.joda.time.{DateTime, LocalDate, YearMonth}

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

  def load_images(
    soho_files_path: Path, year_month: YearMonth,
    soho_source: SOHO, dirTreeCreated: Boolean = true) =
    SOHOLoader.load_images(soho_files_path, year_month, soho_source, dirTreeCreated)

  /**
    * Load X-Ray fluxes averaged over all GOES missions
    *
    * */
  def load_fluxes(
    goes_files_path: Path, year_month: YearMonth,
    goes_source: GOES, dirTreeCreated: Boolean = true) =
    GOESLoader.load_goes_data(
      goes_files_path, year_month,
      goes_source, dirTreeCreated)
      .map(p => {

        val data_low_freq = p._2.map(_._1).filterNot(_.isNaN)
        val data_high_freq = p._2.map(_._2).filterNot(_.isNaN)

        val avg_low_freq = data_low_freq.sum/data_low_freq.length
        val avg_high_freq = data_high_freq.sum/data_high_freq.length

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
  def collate_data(
    year_month: YearMonth)(
    goes_source: GOES,
    goes_data_path: Path,
    goes_aggregation: Int,
    goes_reduce_func: (Stream[(DateTime, (Double, Double))]) => (DateTime, (Double, Double)),
    image_source: SOHO, images_path: Path,
    dt_round_off: (DateTime) => DateTime, dirTreeCreated: Boolean = true) = {

    val proc_goes_data = load_fluxes(
      goes_data_path, year_month,
      goes_source, dirTreeCreated)
      .grouped(goes_aggregation).map(goes_reduce_func).toMap

    val proc_image_data = load_images(
      images_path, year_month,
      image_source, dirTreeCreated)
      .map(p => (dt_round_off(p._1), p._2)).toMap

    proc_image_data.map(kv => {

      val value =
        if (proc_goes_data.contains(kv._1)) Some(proc_goes_data(kv._1))
        else None

      (kv._1, (kv._2, value))})
      .toStream
      .filter(k => k._2._2.isDefined)
      .map(k => (k._1, (k._2._1, k._2._2.get)))
      .sortBy(_._1.getMillis)
  }

  def create_helios_data_set(
    collated_data: Stream[(DateTime, (Path, (Double, Double)))],
    tt_partition: ((DateTime, (Path, (Double, Double)))) => Boolean,
    scaleDownFactor: Int = 4): HeliosDataSet = {

    val scaleDown = 1/math.pow(2, scaleDownFactor)

    val (train_set, test_set) = collated_data.partition(tt_partition)

    var working_set = HeliosDataSet(null, null, null, null)

    train_set.foreach(entry => {
      val (_, (path, data_label)) = entry

      val im = Image.fromPath(path.toNIO)

      val scaled_image = im.copy.scale(scaleDown)

      val im_tensor = dtf.tensor_from(
        "FLOAT32", 1, scaled_image.height, scaled_image.width, 4)(
        scaled_image.argb.toSeq.flatten)

      val label = dtf.tensor_from("FLOAT32", 1, 2)(Seq(data_label._1, data_label._2))

      val trainImages = if(working_set.trainData == null) {
        im_tensor
      } else {
        dtf.concatenate(Seq(working_set.trainData, im_tensor), axis = 0)
      }

      val trainTargets = if(working_set.trainLabels == null) {
        label
      } else {

        dtf.concatenate(Seq(working_set.trainLabels, label), axis = 0)
      }

      working_set = working_set.copy(trainData = trainImages, trainLabels = trainTargets)

    })

    test_set.foreach(entry => {
      val (_, (path, data_label)) = entry

      val im = Image.fromPath(path.toNIO)
      val im_tensor = dtf.tensor_from("FLOAT32", 1, 32, 32, 4)(im.copy.scale(0.0625).argb.toSeq.flatten)

      val label = dtf.tensor_from("FLOAT32", 1, 2)(Seq(data_label._1, data_label._2))

      val testImages = if(working_set.testData == null) {
        im_tensor
      } else {
        dtf.concatenate(Seq(working_set.testData, im_tensor), axis = 0)
      }

      val testTargets = if(working_set.testLabels == null) {
        label
      } else {

        dtf.concatenate(Seq(working_set.testLabels, label), axis = 0)
      }

      working_set = working_set.copy(testData = testImages, testLabels = testTargets)

    })

    working_set
  }

}
