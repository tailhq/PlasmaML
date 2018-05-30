import ammonite.ops._
import org.joda.time._
import com.sksamuel.scrimage._

import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.dtflearn
import io.github.mandar2812.dynaml.pipes._

import _root_.io.github.mandar2812.PlasmaML.helios
import io.github.mandar2812.PlasmaML.helios.core.WeightedTimeSeriesLoss
import io.github.mandar2812.PlasmaML.helios.data.{SOHO, SOHOData}
import io.github.mandar2812.PlasmaML.utils.L2Regularization

import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.api.{::, FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main(
  test_year: Int           = 2003,
  image_source: SOHO       = SOHO(SOHOData.Instruments.MDIMAG, 512),
  re: Boolean              = true,
  time_horizon: (Int, Int) = (18, 56),
  opt: Optimizer           = tf.train.AdaDelta(0.01),
  reg: Double              = 0.001,
  prior_wt: Double         = 0.85,
  error_wt: Double         = 1.0,
  temp: Double             = 0.75,
  maxIt: Int               = 200000,
  miniBatch: Int           = 16,
  tmpdir: Path             = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String          = "mdi_rbfloss_results.csv") = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.generate_data_omni(deltaT = time_horizon)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold = 650.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, Seq[Double]))) =>
  if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max >= sw_threshold) false
  else true

  val dt = DateTime.now()


  val image_sizes = image_source.size

  val crop_solar_image = DataPipe((image: Image) => {

    val image_magic_ratio = 268.0/512.0
    val start = (1.0 - image_magic_ratio)*image_sizes/2
    val patch_size = image_sizes*image_magic_ratio

    image.copy.subimage(start.toInt, start.toInt, patch_size.toInt, patch_size.toInt).scale(0.5)
  })

  val summary_dir_prefix = "swtl_"+image_source.instrument+"_"+image_source.size

  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val summary_dir = summary_dir_prefix+summary_dir_postfix

  val num_pred_dims = 2*data.head._2._2.length

  val output_mapping = WeightedTimeSeriesLoss.output_mapping(
    "Output/ProbWeightedTS",
    data.head._2._2.length)

  val ff_stack_sizes = Seq(128, 64, 50, 30, num_pred_dims)
  val ff_index = 4


  val architecture = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_pyramid(
        size = 2, num_channels_input = 4)(
        start_num_bits = 5, end_num_bits = 3)(
        relu_param = 0.1f, dropout = true,
        keep_prob = 0.6f) >>
      //dtflearn.conv2d_unit(Shape(2, 2, 8, 4), (16, 16), dropout = false)(5) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
      tf.learn.Flatten("Flatten_3") >>
      dtflearn.feedforward_stack(
        (i: Int) => if(i%2 == 0) tf.learn.ReLU("Act_"+i, 0.01f) else dtflearn.Phi("Act_"+i), FLOAT64)(
        ff_stack_sizes,
        starting_index = ff_index)
  } >> output_mapping

  val net_layer_sizes       = Seq(-1) ++ ff_stack_sizes
  val layer_parameter_names = (ff_index until ff_index + ff_stack_sizes.length).map(i => "Linear_"+i+"/Weights")
  val layer_datatypes       = Seq("FLOAT64", "FLOAT64", "FLOAT64")
  val layer_shapes          = net_layer_sizes.sliding(2).toSeq.map(c => Shape(c.head, c.last))

  val loss_func = WeightedTimeSeriesLoss(
    "Loss/ProbWeightedTS",
    data.head._2._2.length,
    prior_wt = prior_wt,
    temperature = temp) >>
    L2Regularization(
      layer_parameter_names,
      layer_datatypes,
      layer_shapes,
      reg)

  helios.run_experiment_omni(
    data, tt_partition, resample = re,
    preprocess_image = crop_solar_image)(
    summary_dir, maxIt, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt)

}
