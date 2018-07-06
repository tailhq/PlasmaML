import ammonite.ops._
import org.joda.time._
import com.sksamuel.scrimage._
import com.sksamuel.scrimage.filter._
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}
import io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.data.SOHO
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Instruments._
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Resolutions._
import io.github.mandar2812.PlasmaML.utils.L2Regularization
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.{::, FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

import scala.util.Random


def median(list: Seq[Byte]): Double = {
  val random: (Int) => Int = Random.nextInt

  def medianK(list_sample: Seq[Byte], k: Int, pivot: Byte): Double = {
    val split_list = list_sample.partition(_ < pivot)
    val s = split_list._1.length

    if(s == k) {
      pivot
    } else if (s == 0 && list_sample.sum == pivot * list_sample.length) {
      pivot
    } else if(s < k) {
      medianK(split_list._2, k - s,
        split_list._2(random(split_list._2.length)))
    } else {
      medianK(split_list._1, k,
        split_list._1(random(split_list._1.length)))
    }
  }

  if(list.length % 2 == 0) {
    val medA = medianK(list, list.length/2, list(random(list.length)))
    val medB = medianK(list, list.length/2 - 1, list(random(list.length)))
    (medA + medB)/2.0
  } else {
    medianK(list, list.length/2, list(random(list.length)))
  }
}

@main
def main(
  year_range: Range             = 2001 to 2004,
  test_year: Int                = 2003,
  image_sources: Seq[SOHO]      = Seq(SOHO(MDIMAG, s512), SOHO(EIT195, s512)),
  re: Boolean                   = true,
  time_horizon: (Int, Int)      = (24, 72),
  time_history: Int             = 8,
  conv_ff_stack_sizes: Seq[Int] = Seq(256, 128, 64, 32),
  hist_ff_stack_sizes: Seq[Int] = Seq(20, 16),
  ff_stack: Seq[Int]            = Seq(128, 64),
  opt: Optimizer                = tf.train.AdaDelta(0.01),
  reg: Double                   = 0.0001,
  prior_wt: Double              = 1.0,
  error_wt: Double              = 4.0,
  temp: Double                  = 2.5,
  stop_criteria: StopCriteria   = dtflearn.max_iter_stop(5000),
  miniBatch: Int                = 16,
  tmpdir: Path                  = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String               = "mdi_rbfloss_results.csv",
  inMemory: Boolean = false) = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.data.generate_data_mc_omni_ext(
    year_range, image_sources,
    deltaT = time_horizon,
    history = time_history)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold   = 750.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: helios.data.MC_PATTERN_EXT) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._2.max >= sw_threshold) false
    else true

  val dt = DateTime.now().withZone(DateTimeZone.forOffsetHours(1))

  val image_sizes = image_sources.head.size

  val crop_solar_image = DataPipe((image: Image) => {

    val image_magic_ratio = 268.0/512.0
    val start = (1.0 - image_magic_ratio)*image_sizes/2
    val patch_size = image_sizes*image_magic_ratio

    val im_copy = if(image.dimensions._1 != s512) {
      image.copy.scaleTo(s512, s512)
    } else {
      image.copy
    }

    im_copy.subimage(start.toInt, start.toInt, patch_size.toInt, patch_size.toInt)
      .scale(0.5)
      .filter(GrayscaleFilter)

  })

  val images_to_byte = DataPipe((is: Seq[Image]) => {

    /*im_byte_coll.flatMap(_.zipWithIndex)
      .groupBy(_._2)
      .mapValues(_.map(_._1))
      .toStream.par
      .map(c => (c._1, median(c._2).toByte))
      .toArray.sortBy(_._1)
      .map(_._2)*/
    is.head.argb.map(_.last.toByte)
  })

  val summary_dir_prefix = "swtl_"+image_sources.map(s => s.instrument+"_"+s.size).mkString("_")

  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val summary_dir = summary_dir_prefix+summary_dir_postfix

  val num_pred_dims = 2*data.head._2._2._2.length

  val output_mapping = helios.learn.cdt_loss.output_mapping(
    "Output/CDT-SW",
    data.head._2._2._2.length)


  /*
  * The following code segment defines the
  * dimensions of the architecture components.
  * */
  val ff_stack_sizes = ff_stack ++ Seq(num_pred_dims)
  val ff_index_conv  = 1
  val ff_index_hist  = ff_index_conv + conv_ff_stack_sizes.length
  val ff_index_fc    = ff_index_hist + hist_ff_stack_sizes.length


  /*
  * =========
  * Module A
  * =========
  *
  * Segment of the architecture which
  * acts on the image pixels.
  *
  * Consists of a CNN stack which
  * feeds into a feed-forward stack
  * followed by an Upwind 1d propagation
  * model.
  * */
  val image_neural_stack = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_unit(Shape(4, 4, image_sources.length, 64), dropout = false)(0) >>
      dtflearn.conv2d_unit(Shape(2, 2, 64, 32), dropout = false)(1) >>
      tf.learn.MaxPool("MaxPool_1", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      dtflearn.conv2d_unit(Shape(2, 2, 32, 16), dropout = false)(2) >>
      dtflearn.conv2d_unit(Shape(2, 2, 16, 8), dropout = false)(3) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      tf.learn.Flatten("Flatten_3") >>
      dtflearn.feedforward_stack(
        get_act = i => dtflearn.Phi("Act_"+i),
        dataType = FLOAT64)(
        layer_sizes = conv_ff_stack_sizes,
        starting_index = ff_index_conv) >>
      helios.learn.upwind_1d("Upwind1d", (30.0, 215.0), 50) >>
      tf.learn.Flatten("Flatten_4")
  }

  /*
  * =========
  * Module B
  * =========
  *
  * The following stack acts on the time history
  * of the solar wind. It consists of a sequence
  * of feed-forward layers.
  * */
  val omni_history_stack = {
    tf.learn.Cast("Input/Cast", FLOAT64) >>
      dtflearn.feedforward_stack(
        (i: Int) => if(i%2 == 1) tf.learn.ReLU("Act_"+i, 0.01f) else dtflearn.Phi("Act_"+i),
        FLOAT64)(
        hist_ff_stack_sizes,
        starting_index = ff_index_hist)
  }

  /*
  * =========
  * Module C
  * =========
  *
  * The component of the prediction architecture which acts
  * on the combination of the features produced by
  * modules A and B
  * */
  val fc_stack = dtflearn.feedforward_stack(
    (i: Int) => dtflearn.Phi("Act_"+i),
    FLOAT64)(
    ff_stack_sizes,
    starting_index = ff_index_fc)


  val architecture = dtflearn.tuple2_layer("OmniCTLStack", image_neural_stack, omni_history_stack) >>
    dtflearn.concat_tuple2("StackFeatures", axis = 1) >>
    fc_stack >>
    output_mapping

  val (layer_shapes_conv, layer_parameter_names_conv, layer_datatypes_conv) =
    dtfutils.get_ffstack_properties(Seq(-1) ++ conv_ff_stack_sizes, ff_index_conv)

  val (layer_shapes_hist, layer_parameter_names_hist, layer_datatypes_hist) =
    dtfutils.get_ffstack_properties(Seq(-1) ++ hist_ff_stack_sizes, ff_index_hist)

  val (layer_shapes_fc, layer_parameter_names_fc, layer_datatypes_fc) =
    dtfutils.get_ffstack_properties(Seq(-1) ++ ff_stack_sizes, ff_index_fc)


  val loss_func = helios.learn.cdt_loss(
    "Loss/CDT-SW",
    data.head._2._2._2.length,
    prior_wt = prior_wt,
    temperature = temp) >>
    L2Regularization(
      layer_parameter_names_conv ++ layer_parameter_names_hist ++ layer_parameter_names_fc,
      layer_datatypes_conv ++ layer_datatypes_hist ++ layer_datatypes_fc,
      layer_shapes_conv ++ layer_shapes_hist ++ layer_shapes_fc,
      reg)

  helios.run_experiment_mc_omni_ext(
    image_sources,
    data, tt_partition, resample = re,
    image_sources.map(s => (s, crop_solar_image)).toMap,
    images_to_byte)(
    summary_dir,
    stop_criteria, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt,
    inMemoryModel = inMemory)

}
