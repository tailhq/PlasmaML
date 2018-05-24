import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import io.github.mandar2812.PlasmaML.helios.core.WeightedTimeSeriesLoss
import io.github.mandar2812.PlasmaML.helios.data.{SOHO, SOHOData}
import io.github.mandar2812.PlasmaML.utils.L2Regularization
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.dtflearn
import org.joda.time._
import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.api.{::, FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main(
  test_year: Int           = 2003,
  image_source: SOHO       = SOHO(SOHOData.Instruments.MDIMAG, 512),
  re: Boolean              = true,
  sc_down: Int             = 1,
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

  val summary_dir_prefix = "omni_swtl_"+image_source.instrument+"_"+image_source.size+"_"+sc_down

  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  val summary_dir = summary_dir_prefix+summary_dir_postfix


  val num_pred_dims = 2*data.head._2._2.length

  val architecture = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_pyramid(
        size = 2, num_channels_input = 4)(
        start_num_bits = 7, end_num_bits = 3)(
        relu_param = 0.1f, dropout = true,
        keep_prob = 0.6f) >>
      dtflearn.conv2d_unit(Shape(2, 2, 8, 4), (16, 16), dropout = false)(4) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
      tf.learn.Flatten("Flatten_3") >>
      tf.learn.Linear("FC_Layer_4", 256) >>
      dtflearn.Phi("Act_4") >>
      tf.learn.Linear("FC_Layer_5", 128) >>
      dtflearn.Phi("Act_5") >>
      tf.learn.Linear("FC_Layer_6", num_pred_dims)
  }

  val net_layer_sizes       = Seq(-1, 256, 128, num_pred_dims)
  val layer_parameter_names = Seq(4, 5, 6).map(i => "FC_Layer_"+i+"/Weights")
  val layer_datatypes       = Seq("FLOAT32", "FLOAT64", "FLOAT64")
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
    data, tt_partition, resample = re, scaleDown = sc_down)(
    summary_dir, maxIt, tmpdir,
    arch = architecture,
    lossFunc = loss_func,
    optimizer = opt)

}
