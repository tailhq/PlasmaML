import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import io.github.mandar2812.PlasmaML.dynamics.mhd.{UpwindPropogate, UpwindTF}
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.tensorflow.dtflearn
import io.github.mandar2812.dynaml.tensorflow.layers.FiniteHorizonCTRNN
import org.joda.time._
import org.platanios.tensorflow.api.{FLOAT32, FLOAT64, Shape, tf}
import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer

@main
def main(
  test_year: Int = 2003,
  re: Boolean = true,
  opt: Optimizer = tf.train.AdaDelta(0.01),
  maxIt: Int = 200000,
  tmpdir: Path = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String = "mdi_rbfloss_results.csv") = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.generate_data_omni()

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold = 650.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, Seq[Double]))) =>
  if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max >= sw_threshold) false
  else true

  val summary_dir = if(re) "mdi_resample_"+test_year else "mdi_"+test_year

  val architecture = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_pyramid(2, 4)(6, 3)(0.01f, dropout = true, 0.4f) >>
      dtflearn.conv2d_unit(Shape(2, 2, 8, 4), (16, 16), dropout = false, relu_param = 0.01f)(4) >>
      tf.learn.MaxPool("MaxPool_5", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
      tf.learn.Flatten("Flatten_5") >>
      dtflearn.feedforward(64)(6) >>
      tf.learn.SELU("SELU_6") >>
      dtflearn.feedforward(16)(7) >>
      tf.learn.Sigmoid("Sigmoid_7") >>
      dtflearn.feedforward(4)(8) >>
      FiniteHorizonCTRNN("fhctrnn_9", 4, 5, 0.2d) >>
      tf.learn.Flatten("Flatten_9") >>
      tf.learn.Linear("OutputLayer", 2)
  }

  helios.run_experiment_omni(
    data, tt_partition, resample = re)(
    summary_dir, maxIt, tmpdir, arch = architecture)

}
