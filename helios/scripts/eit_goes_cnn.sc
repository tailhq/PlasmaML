import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data._
import ammonite.ops._
import com.sksamuel.scrimage.Image
import org.joda.time._

import org.platanios.tensorflow.api._


val data_dir_name = "tmp"//"data_repo"

val data_dir = home/'tmp/'helios
val soho_dir = data_dir/'soho
val goes_dir = data_dir/'goes

val (year, month, day) = ("2003", "10", "28")

val halloween_start = new DateTime(2003, 10, 28, 8, 59)

/*
val halloween_images = ls! soho_dir |? (_.isDir) ||
  (d => ls! d |? (_.segments.contains(year)) || (ls! _) |? (_.segments.contains(month))) ||
  (ls! _ )

val images_mdi = halloween_images |?
  (_.segments.contains(SOHOData.Instruments.MDIMAG)) |?
  (_.segments.last.contains(year+month+day))

val images_mdiigr = halloween_images |?
  (_.segments.contains(SOHOData.Instruments.MDIIGR)) |?
  (_.segments.last.contains(year+month+day))

val images_eit171 = halloween_images |?
  (_.segments.contains(SOHOData.Instruments.EIT171)) |?
  (_.segments.last.contains(year+month+day))

val images_eit195 = halloween_images |?
  (_.segments.contains(SOHOData.Instruments.EIT195)) |?
  (_.segments.last.contains(year+month+day))

val images_eit284 = halloween_images |?
  (_.segments.contains(SOHOData.Instruments.EIT284)) |?
  (_.segments.last.contains(year+month+day))

val images_eit304 = halloween_images |?
  (_.segments.contains(SOHOData.Instruments.EIT304)) |?
  (_.segments.last.contains(year+month+day))


val halloween_goes_flux_files = ls! goes_dir/'xrs_5m/year/month

val halloween_flux_file = goes_dir/'xrs_5m/year/month/("g10_xrs_5m_"+year+month+"01_"+year+month+"31.csv")
*/

val goes_data = GOESLoader.load_goes_data(
  goes_dir, new YearMonth(year.toInt, month.toInt),
  GOES(GOESData.Quantities.XRAY_FLUX_5m))

/*val mdi_images = SOHOLoader.load_images(
  soho_dir, new YearMonth(year.toInt, month.toInt),
  SOHO(SOHOData.Instruments.MDIMAG, 512))*/

val eit195_images = SOHOLoader.load_images(
  soho_dir, new YearMonth(year.toInt, month.toInt),
  SOHO(SOHOData.Instruments.EIT195, 512))

//val halloween_mdi_data = mdi_images.filter(p => p._1.isAfter(halloween_start))

val halloween_eit195_data = eit195_images.filter(p => p._1.isAfter(halloween_start))

val avgd_goes_data = goes_data.map(p => {

  val data_low_freq = p._2.map(_._1).filterNot(_.isNaN)
  val data_high_freq = p._2.map(_._2).filterNot(_.isNaN)

  val avg_low_freq = data_low_freq.sum/data_low_freq.length
  val avg_high_freq = data_high_freq.sum/data_high_freq.length

  (p._1, (avg_low_freq, avg_high_freq))
})

val halloween_goes_data = avgd_goes_data.filter(
  p => p._1.isAfter(halloween_start))
  .grouped(2).map(gr => (gr.head._1, gr.map(_._2).max))
  .toMap

val halloween_eit195_proc = halloween_eit195_data.map(p => {
  val minutes: Int = p._1.getMinuteOfHour/10

  (new DateTime(
    p._1.getYear, p._1.getMonthOfYear,
    p._1.getDayOfMonth, p._1.getHourOfDay,
    minutes*10),
    p._2)
}).toMap

val collated_data = halloween_eit195_proc.map(kv => {

  val value =
    if (halloween_goes_data.contains(kv._1)) Some(halloween_goes_data(kv._1))
    else None

  (kv._1, (kv._2, value))})
  .toSeq
  .filter(k => k._2._2.isDefined)
  .map(k => (k._1, (k._2._1, k._2._2.get)))
  .sortBy(_._1.getMillis)

//val image_path = collated_data.head._2._1

//val example_eit = Image.fromFile(new java.io.File(image_path.toString()))

var data = HeliosDataSet(null, null, null, null)

//val image_tensor = dtf.tensor_from("UINT16", 1, 512, 512, 4)(example_eit.argb.toStream.flatten)

collated_data.foreach(entry => {
  val (_, (path, data_label)) = entry

  val im = Image.fromPath(path.toNIO)
  val im_tensor = dtf.tensor_from("UINT8", 1, 32, 32, 4)(im.copy.scale(0.0625).argb.toSeq.flatten)

  val label = dtf.tensor_from("FLOAT64", 1, 2)(Seq(data_label._1, data_label._2))

  val trainImages = if(data.trainData == null) {
    im_tensor
  } else {
    dtf.concatenate(Seq(data.trainData, im_tensor), axis = 0)
  }

  val trainTargets = if (data.trainLabels == null) {
    label
  } else {

    dtf.concatenate(Seq(data.trainLabels, label), axis = 0)
  }

  data = data.copy(trainData = trainImages, trainLabels = trainTargets)

})

val trainImages = tf.data.TensorSlicesDataset(data.trainData)

val trainLabels = tf.data.TensorSlicesDataset(data.trainLabels)

val trainData =
  trainImages.zip(trainLabels)
    .repeat()
    .shuffle(10000)
    .batch(32)
    .prefetch(10)
