import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import io.github.mandar2812.PlasmaML.helios.core.WeightedL2FluxLoss
import io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time._

@main
def main(
  test_year: Int = 2003,
  longWL: Boolean = false,
  re: Boolean = true,
  tmpdir: Path = home/"tmp",
  resFile: String = "mdi_ext_wtloss_results.csv") = {

  //Data with MDI images

  val results = if(longWL) "longwl_"+resFile else resFile

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.generate_data_goes()

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val flux_threshold = if(longWL) -5.5d else -6.5d

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, (Double, Double)))) =>
    if(p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._1 >= flux_threshold) false
    else true

  val summary_dir = if(re) "mdi_wtloss_ext_resample_"+test_year else "mdi_wtloss_ext_"+test_year

  val res = helios.run_experiment_goes(
    data, tt_partition, resample = re,
    longWavelength = longWL)(
    summary_dir, 120000, tmpdir,
    lossFunc = new WeightedL2FluxLoss("Loss/WeightedL2FluxLoss"))


  //Write the cross validation score in a results file

  val accuracy = res._3

  if(!exists(tmpdir/results)) write(tmpdir/resFile, "testyear,accuracy\n")

  write.append(tmpdir/results, s"$test_year,$accuracy\n")

  pprint.pprintln(res)
}