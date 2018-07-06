import ammonite.ops._
import org.joda.time._
import com.sksamuel.scrimage._
import com.sksamuel.scrimage.filter.GrayscaleFilter
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.data.{SDO, SOHO, SOHOData, SolarImagesSource}
import io.github.mandar2812.PlasmaML.helios.data.SDOData.Instruments._
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Instruments._
import io.github.mandar2812.PlasmaML.utils.L2Regularization
import io.github.mandar2812.dynaml.DynaMLPipe
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.{::, FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer


@main
def main[T <: SolarImagesSource](
  year_range: Range             = 2001 to 2004,
  test_year: Int                = 2003,
  image_source: T               = SOHO(SOHOData.Instruments.MDIMAG, 512),
  re: Boolean                   = true,
  time_horizon: (Int, Int)      = (18, 56),
  time_history: Int             = 8,
  conv_ff_stack_sizes: Seq[Int] = Seq(256, 128, 64, 32, 8),
  hist_ff_stack_sizes: Seq[Int] = Seq(32, 16),
  ff_stack: Seq[Int]            = Seq(80, 64),
  opt: Optimizer                = tf.train.AdaDelta(0.01),
  reg: Double                   = 0.001,
  prior_wt: Double              = 0.85,
  error_wt: Double              = 1.0,
  temp: Double                  = 0.75,
  stop_criteria: StopCriteria   = dtflearn.max_iter_stop(5000),
  miniBatch: Int                = 16,
  tmpdir: Path                  = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String               = "mdi_rbfloss_results.csv") = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.data.generate_data_omni_ext[T](
    year_range, image_source,
    deltaT = time_horizon,
    history = time_history)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold   = 700.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, (Seq[Double], Seq[Double])))) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._2.max >= sw_threshold) false
    else true

  val dt = DateTime.now()


  val (image_sizes, magic_ratio) = image_source match {
    case SOHO(_, s) => (s, 268.0/512.0)
    case SDO(_, s)  => (s, 333.0/512.0)
  }

  val image_preprocess =
    helios.data.image_central_patch(magic_ratio, image_sizes) >
      DataPipe((i: Image) => i.copy.scale(scaleFactor = 0.5))

  val (image_filter, num_channels, image_to_byte) = image_source match {
    case _: SOHO => (
      DataPipe((i: Image) => i.filter(GrayscaleFilter)), 1,
      DataPipe((i: Image) => i.argb.map(_.last.toByte)))

    case SDO(AIA094335193, s) => (
      DynaMLPipe.identityPipe[Image], 4,
      DataPipe((i: Image) => i.argb.flatten.map(_.toByte)))

    case _: SDO  => (
      DynaMLPipe.identityPipe[Image], 1,
      DataPipe((i: Image) => i.argb.map(_.last.toByte)))
  }


  val summary_dir_prefix = "swtl_"+image_source.toString

  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val summary_dir = summary_dir_prefix+summary_dir_postfix

  val num_pred_dims = 2*data.head._2._2._2.length

  val output_mapping = helios.learn.cdt_loss.output_mapping(
    "Output/CDT-SW",
    data.head._2._2._2.length)


  val ff_stack_sizes = ff_stack ++ Seq(num_pred_dims)

  val ff_index_conv  = 1

  val ff_index_hist  = ff_index_conv + conv_ff_stack_sizes.length
  val ff_index_fc    = ff_index_hist + hist_ff_stack_sizes.length

  val image_neural_stack = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_pyramid(
        size = 2, num_channels_input = num_channels)(
        start_num_bits = 5, end_num_bits = 3)(
        relu_param = 0.1f, dropout = false,
        keep_prob = 0.6f) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      tf.learn.Flatten("Flatten_3") >>
      dtflearn.feedforward_stack(
        (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
        conv_ff_stack_sizes,
        starting_index = ff_index_conv) >>
      tf.learn.Cast("Cast/Float", FLOAT32) >>
      helios.learn.upwind_1d("Upwind1d", (30.0, 215.0), 20) >>
      tf.learn.Flatten("Flatten_4")
  }

  val omni_history_stack = {
    tf.learn.Cast("Input/Cast", FLOAT64) >>
      dtflearn.feedforward_stack(
        (i: Int) => dtflearn.Phi("Act_"+i),
        FLOAT64)(
        hist_ff_stack_sizes,
        starting_index = ff_index_hist)
  }

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

  helios.run_experiment_omni_ext(
    data, tt_partition, resample = re,
    preprocess_image = image_preprocess > image_filter,
    image_to_bytearr = image_to_byte,
    num_channels_image = num_channels)(
    summary_dir, stop_criteria, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt)

}
