import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time._
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.tensorflow.dtflearn
import org.platanios.tensorflow.api.ops.NN.SameConvPadding

@main
def main(
  test_year: Int = 2003,
  longWL: Boolean = false,
  re: Boolean = true,
  tmpdir: Path = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String = "mdi_ext_wtloss_results.csv",
  maxIt: Int = 200000) = {

  //Data with MDI images

  val results = if(longWL) "longwl_"+resFile else resFile

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.data.generate_data_goes()

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val flux_threshold = if(longWL) -5.5d else -6.5d

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = if(longWL) {

    (p: (DateTime, (Path, (Double, Double)))) =>
      if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._2 >= flux_threshold) false
      else true

  } else {

      (p: (DateTime, (Path, (Double, Double)))) =>
        if(p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._1 >= flux_threshold) false
        else true

  }

  val summary_dir = if(re) "mdi_wtloss_ext_resample_"+test_year else "mdi_wtloss_ext_"+test_year

  val architecture = {
    tf.learn.Cast("Input/Cast", FLOAT32) >>
      dtflearn.conv2d_pyramid(2, 4)(6, 3)(0.01f, dropout = true, 0.6f) >>
      tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      dtflearn.conv2d_unit(Shape(2, 2, 8, 4), (16, 16), dropout = false, relu_param = 0.01f)(4) >>
      tf.learn.MaxPool("MaxPool_5", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
      tf.learn.Flatten("Flatten_5") >>
      dtflearn.feedforward(128)(6) >>
      dtflearn.Tanh("Tanh_6") >>
      dtflearn.feedforward(64)(7) >>
      dtflearn.Tanh("Tanh_7") >>
      dtflearn.feedforward(32)(8) >>
      dtflearn.Tanh("Tanh_6") >>
      tf.learn.Linear("OutputLayer", 1)
  }

  val res = helios.run_experiment_goes(
    data, tt_partition, resample = re,
    longWavelength = longWL)(
    summary_dir, maxIt, tmpdir,
    arch = architecture,
    lossFunc = helios.learn.weightedL2FluxLoss("Loss/WeightedL2FluxLoss"))


  //Write the cross validation score in a results file

  val accuracy = res._3.results(0).sum()

  if(!exists(tmpdir/results)) write(tmpdir/results, "testyear,accuracy\n")

  write.append(tmpdir/results, s"$test_year,$accuracy\n")

  pprint.pprintln(res)

  res
}
