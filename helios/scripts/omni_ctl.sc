import ammonite.ops._
import org.joda.time._

import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}

import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data
import io.github.mandar2812.PlasmaML.helios.data.{SDO, SOHO, SOHOData, SolarImagesSource}
import io.github.mandar2812.PlasmaML.helios.data.SDOData.Instruments._
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Instruments._
import io.github.mandar2812.PlasmaML.utils.L2Regularization

import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.{FLOAT32, FLOAT64, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main[T <: SolarImagesSource](
  year_range: Range             = 2001 to 2004,
  test_year: Int                = 2003,
  ff_stack_sizes: Seq[Int]      = Seq(256, 128, 64),
  image_source: T               = SOHO(MDIMAG, 512),
  buffer_size: Int              = 2000,
  re: Boolean                   = true,
  time_horizon: (Int, Int)      = (18, 56),
  image_hist: Int               = 0,
  image_hist_downsamp: Int      = 1,
  opt: Optimizer                = tf.train.AdaDelta(0.01),
  reg: Double                   = 0.001,
  prior_wt: Double              = 0.85,
  error_wt: Double              = 1.0,
  temp: Double                  = 0.75,
  stop_criteria: StopCriteria   = dtflearn.max_iter_stop(5000),
  miniBatch: Int                = 16,
  tmpdir: Path                  = root/"home"/System.getProperty("user.name")/"tmp",
  path_to_images: Option[Path]  = None,
  existingModelDir: String      = "") = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  data.buffer_size_(buffer_size)

  val dataset = data.generate_data_omni[T](
    year_range, image_source,
    deltaT = time_horizon,
    images_data_dir = path_to_images)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold   = 700.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)
  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, Seq[Double]))) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max >= sw_threshold) false
    else true


  val (image_sizes, magic_ratio) = image_source match {
    case SOHO(_, s) => (s, 268.0/512.0)
    case SDO(_, s)  => (s, 333.0/512.0)
  }

  val (image_filter, num_channels, image_to_byte) = data.image_process_metadata(image_source)

  val patch_range = data.get_patch_range(magic_ratio, image_sizes/2)

  val image_preprocess = data.image_central_patch(magic_ratio, image_sizes) > data.image_scale(0.5)

  //Set the path of the summary directory
  val summary_dir_prefix  = "swtl_"+image_source.toString
  val dt                  = DateTime.now()
  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val (summary_dir , reuse): (String, Boolean)  =
    if(existingModelDir.isEmpty) (summary_dir_prefix+summary_dir_postfix, false)
    else (existingModelDir, true)

  if(reuse) println("\nReusing existing model in directory: "+existingModelDir+"\n")

  //Set some meta-information for the prediction architecture
  val num_pred_dims    = 2*dataset.data.head._2._2.length
  val ff_index         = 6
  val ff_stack         = ff_stack_sizes :+ num_pred_dims

  /*
  * Construct architecture components.
  *
  * 1) Convolutional segment: Consists of two convolutional pyramids interspersed by batch normalisation
  * 2) Pre Upwind feed-forward segment.
  * 3) Upwind 1D computational layer.
  * 4) Post Upwind feed-forward segment.
  * 5) A post processing output mapping.
  * */

  val filter_depths = Seq(
    Seq.fill(4)(20),
    Seq.fill(4)(15),
    Seq.fill(4)(10),
    Seq.fill(4)(5),
    Seq.fill(4)(1)
  )

  val conv_section =
    dtflearn.inception_unit(num_channels*(image_hist_downsamp + 1), filter_depths.head)(layer_index = 1) >>
      dtflearn.inception_unit(filter_depths.head.sum, filter_depths(1))(layer_index = 2) >>
      dtflearn.inception_unit(filter_depths(1).sum, filter_depths(2))(layer_index = 3) >>
      dtflearn.inception_unit(filter_depths(2).sum, filter_depths(3))(layer_index = 4) >>
      dtflearn.inception_unit(filter_depths(3).sum, filter_depths(4))(layer_index = 5)

  val post_conv_ff_stack = dtflearn.feedforward_stack(
    (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
    ff_stack, starting_index = ff_index)

  val output_mapping = helios.learn.cdt_loss.output_mapping(
    "Output/CDT-SW",
    dataset.data.head._2._2.length)


  val architecture = tf.learn.Cast("Input/Cast", FLOAT32) >>
    conv_section >>
    tf.learn.Flatten("Flatten_1") >>
    post_conv_ff_stack >>
    output_mapping


  val (layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(Seq(-1) ++ ff_stack, ff_index)

  val loss_func = helios.learn.cdt_loss(
    "Loss/CDT-SW",
    dataset.data.head._2._2.length,
    prior_wt = prior_wt,
    temperature = temp) >>
    L2Regularization(
      layer_parameter_names,
      layer_datatypes,
      layer_shapes,
      reg)

  helios.run_experiment_omni(
    dataset, tt_partition, resample = re,
    preprocess_image = image_preprocess > image_filter,
    image_to_bytearr = image_to_byte,
    num_channels_image = num_channels,
    image_history = image_hist,
    image_history_downsampling = image_hist_downsamp,
    processed_image_size = (patch_range.length, patch_range.length))(
    summary_dir, stop_criteria, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt,
    reuseExistingModel = reuse)

}
