import ammonite.ops._
import org.joda.time._
import com.sksamuel.scrimage._
import com.sksamuel.scrimage.filter._
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.dtflearn
import io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.data.SOHO
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Instruments._
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Resolutions._
import io.github.mandar2812.PlasmaML.utils.L2Regularization
import org.platanios.tensorflow.api.ops.NN.SamePadding
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

def get_ffstack_properties(neuron_counts: Seq[Int], ff_index: Int): (Seq[Shape], Seq[String], Seq[String]) = {

  val layer_parameter_names = (ff_index until ff_index + neuron_counts.length - 1).map(i => "Linear_"+i+"/Weights")
  val layer_shapes          = neuron_counts.sliding(2).toSeq.map(c => Shape(c.head, c.last))
  val layer_datatypes       = Seq.fill(layer_shapes.length)("FLOAT64")


  (layer_shapes, layer_parameter_names, layer_datatypes)
}


@main
def main(
  test_year: Int                = 2003,
  start_year: Int               = 2001,
  end_year: Int                 = 2006,
  image_sources: Seq[SOHO]      = Seq(SOHO(MDIMAG, s512), SOHO(EIT171, s512)),
  re: Boolean                   = true,
  time_horizon: (Int, Int)      = (18, 56),
  time_history: Int             = 8,
  conv_ff_stack_sizes: Seq[Int] = Seq(256, 128),
  hist_ff_stack_sizes: Seq[Int] = Seq(32, 16),
  ff_stack: Seq[Int]            = Seq(80, 64),
  opt: Optimizer                = tf.train.AdaDelta(0.01),
  reg: Double                   = 0.001,
  prior_wt: Double              = 0.85,
  error_wt: Double              = 1.0,
  temp: Double                  = 0.75,
  maxIt: Int                    = 200000,
  miniBatch: Int                = 16,
  tmpdir: Path                  = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String               = "mdi_rbfloss_results.csv") = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.generate_data_mc_omni_ext(
    year_start = start_year, year_end = end_year, image_sources,
    deltaT = time_horizon, history = time_history)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold = 650.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: helios.MC_PATTERN_EXT) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._2.max >= sw_threshold) false
    else true

  val dt = DateTime.now()


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

    //val num_pixels = is.head.dimensions._1*is.head.dimensions._2

    val im_byte_coll = is.map(_.argb.map(_.last.toByte))

    /*(0 until num_pixels).map(pixel_index => {
      val pixel_sample_for_index = im_byte_coll.map(_(pixel_index))
      median(pixel_sample_for_index.toStream).toByte
    }).toArray*/

    im_byte_coll.flatMap(_.zipWithIndex)
      .groupBy(_._2)
      .mapValues(_.map(_._1))
      .toStream.par
      .map(c => (c._1, median(c._2).toByte))
      .toArray.map(_._2)
  })

  val summary_dir_prefix = "swtl_"+image_sources.map(s => s.instrument+"_"+s.size).mkString("_")

  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val summary_dir = summary_dir_prefix+summary_dir_postfix

  val num_pred_dims = 2*data.head._2._2._2.length

  val output_mapping = helios.learn.cdt_loss.output_mapping(
    "Output/ProbWeightedTS",
    data.head._2._2._2.length)


  val ff_stack_sizes      = ff_stack ++ Seq(num_pred_dims)

  val ff_index_conv = 1

  val ff_index_hist = ff_index_conv + conv_ff_stack_sizes.length
  val ff_index_fc   = ff_index_hist + hist_ff_stack_sizes.length

  val image_neural_stack = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_pyramid(
        size = 2, num_channels_input = image_sources.length)(
        start_num_bits = 5, end_num_bits = 3)(
        relu_param = 0.1f, dropout = false,
        keep_prob = 0.6f) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
      tf.learn.Flatten("Flatten_3") >>
      dtflearn.feedforward_stack(
        (i: Int) => if(i%2 == 1) tf.learn.ReLU("Act_"+i, 0.01f) else dtflearn.Phi("Act_"+i), FLOAT64)(
        conv_ff_stack_sizes,
        starting_index = ff_index_conv)
  }

  val omni_history_stack = {
    tf.learn.Cast("Input/Cast", FLOAT64) >>
      dtflearn.feedforward_stack(
        (i: Int) => if(i%2 == 1) tf.learn.ReLU("Act_"+i, 0.01f) else dtflearn.Phi("Act_"+i),
        FLOAT64)(
        hist_ff_stack_sizes,
        starting_index = ff_index_hist)
  }

  val fc_stack = dtflearn.feedforward_stack(
    (i: Int) => if(i%2 == 1) tf.learn.ReLU("Act_"+i, 0.01f) else dtflearn.Phi("Act_"+i),
    FLOAT64)(
    ff_stack_sizes,
    starting_index = ff_index_fc)


  val architecture = dtflearn.tuple2_layer("OmniCTLStack", image_neural_stack, omni_history_stack) >>
    dtflearn.concat_tuple2("StackFeatures", axis = 1) >>
    fc_stack >>
    output_mapping

  val (layer_shapes_conv, layer_parameter_names_conv, layer_datatypes_conv) =
    get_ffstack_properties(Seq(-1) ++ conv_ff_stack_sizes, ff_index_conv)

  val (layer_shapes_hist, layer_parameter_names_hist, layer_datatypes_hist) =
    get_ffstack_properties(Seq(-1) ++ hist_ff_stack_sizes, ff_index_hist)

  val (layer_shapes_fc, layer_parameter_names_fc, layer_datatypes_fc) =
    get_ffstack_properties(Seq(-1) ++ ff_stack_sizes, ff_index_fc)


  val loss_func = helios.learn.cdt_loss(
    "Loss/ProbWeightedTS",
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
    summary_dir, maxIt, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt)

}
