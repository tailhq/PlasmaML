import ammonite.ops._
import org.joda.time._
import com.sksamuel.scrimage._
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}
import io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.PlasmaML.helios
import com.sksamuel.scrimage.filter.GrayscaleFilter
import io.github.mandar2812.PlasmaML.helios.core.CausalDynamicTimeLag
import io.github.mandar2812.PlasmaML.helios.data.{SDO, SOHO, SOHOData, SolarImagesSource}
import io.github.mandar2812.PlasmaML.helios.data.SDOData.Instruments._
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Instruments._
import io.github.mandar2812.PlasmaML.utils.L2Regularization
import io.github.mandar2812.dynaml.DynaMLPipe
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.{FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main[T <: SolarImagesSource](
  year_range: Range             = 2001 to 2004,
  test_year: Int                = 2003,
  pre_upwind_ff_sizes: Seq[Int] = Seq(256, 128, 64, 32),
  ff_stack_sizes: Seq[Int]      = Seq(256, 128, 64),
  image_source: T               = SOHO(MDIMAG, 512),
  buffer_size: Int              = 2000,
  re: Boolean                   = true,
  time_horizon: (Int, Int)      = (18, 56),
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

  helios.data.buffer_size_(buffer_size)

  val data           = helios.data.generate_data_omni[T](year_range, image_source, deltaT = time_horizon)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold   = 700.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, Seq[Double]))) =>
  if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max >= sw_threshold) false
  else true

  val dt = DateTime.now()


  val (image_sizes, magic_ratio) = image_source match {
    case SOHO(_, s) => (s, 268.0/512.0)
    case SDO(_, s)  => (s, 333.0/512.0)
  }

  val image_preprocess =
    helios.data.image_central_patch(magic_ratio, image_sizes) >
      DataPipe((i: Image) => i.copy.scale(scaleFactor = 0.5))


  val (image_filter, num_channels, image_to_byte) = helios.data.image_process_metadata(image_source)

  val summary_dir_prefix = "swtl_"+image_source.toString

  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val summary_dir = summary_dir_prefix+summary_dir_postfix

  val num_pred_dims = 2*data.head._2._2.length

  val output_mapping = CausalDynamicTimeLag.output_mapping(
    "Output/CDT-SW",
    data.head._2._2.length)

  val pre_upwind_index = 4
  val ff_index         = pre_upwind_index + pre_upwind_ff_sizes.length
  val ff_stack         = ff_stack_sizes :+ num_pred_dims


  val architecture =
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_unit(Shape(4, 4, num_channels, 20), dropout = true)(0) >>
      dtflearn.conv2d_unit(Shape(2, 2, 20, 15), dropout = true)(1) >>
      tf.learn.MaxPool("MaxPool_1", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      dtflearn.conv2d_unit(Shape(2, 2, 15, 10), dropout = true)(2) >>
      dtflearn.conv2d_unit(Shape(2, 2, 10, 8), dropout = true)(3) >>
      tf.learn.MaxPool("MaxPool_2", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      tf.learn.Flatten("Flatten_3") >>
      dtflearn.feedforward_stack(
        (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
        pre_upwind_ff_sizes, starting_index = pre_upwind_index) >>
      tf.learn.Cast("Cast/Float", FLOAT32) >>
      helios.learn.upwind_1d("Upwind1d", (30.0, 215.0), 50) >>
      tf.learn.Flatten("Flatten_4") >>
      dtflearn.feedforward_stack(
        (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
        ff_stack, starting_index = ff_index) >>
      output_mapping


  val (layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(Seq(-1) ++ ff_stack, ff_index)

  val (pre_upwind_layer_shapes, pre_upwind_layer_parameter_names, pre_upwind_layer_datatypes) =
    dtfutils.get_ffstack_properties(Seq(-1) ++ pre_upwind_ff_sizes, pre_upwind_index)

  val loss_func = CausalDynamicTimeLag(
    "Loss/CDT-SW",
    data.head._2._2.length,
    prior_wt = prior_wt,
    temperature = temp) >>
    L2Regularization(
      pre_upwind_layer_parameter_names ++ layer_parameter_names,
      pre_upwind_layer_datatypes ++ layer_datatypes,
      pre_upwind_layer_shapes ++ layer_shapes,
      reg)

  helios.run_experiment_omni(
    data, tt_partition, resample = re,
    preprocess_image = image_preprocess > image_filter,
    image_to_bytearr = image_to_byte,
    num_channels_image = num_channels)(
    summary_dir, stop_criteria, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt)

}
