import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import io.github.mandar2812.PlasmaML.helios.core.WeightedTimeSeriesLoss
import io.github.mandar2812.PlasmaML.utils.L2Regularization
import io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time._
import org.platanios.tensorflow.api.{FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main(
  test_year: Int = 2003,
  re: Boolean = true, sc_down: Int = 2,
  time_horizon: (Int, Int) = (18, 56),
  opt: Optimizer = tf.train.AdaDelta(0.01),
  reg: Double = 0.001,
  prior_wt: Double = 0.85,
  error_wt: Double = 1.0,
  temp: Double = 1.0,
  maxIt: Int = 200000,
  tmpdir: Path = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String = "mdi_rbfloss_results.csv") = {

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

  val summary_dir = if(re) "mdi_resample_"+test_year else "mdi_"+test_year

  val architecture = helios.learn.cnn_sw_v2(data.head._2._2.length)

  val net_layer_sizes       = Seq(128, 64, data.head._2._2.length)
  val layer_parameter_names = Seq(4, 5, 6).map(i => "FC_layer_"+i)
  val layer_datatypes       = Seq.fill(layer_parameter_names.length)("FLOAT32")
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
    lossFunc = loss_func)

}
