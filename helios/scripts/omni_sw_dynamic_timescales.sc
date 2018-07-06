import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time._

@main
def main(
  test_year: Int = 2003,
  re: Boolean = true,
  tmpdir: Path = root/"home"/System.getProperty("user.name")/"tmp",
  resFile: String = "mdi_rbfloss_dynamic_results.csv") = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  val data           = helios.data.generate_data_omni(2001 to 2003)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold = 650.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)

  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Path, Seq[Double]))) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max >= sw_threshold) false
    else true

  val summary_dir = if(re) "mdi_dynamic_resample_"+test_year else "mdi_dynamic_"+test_year

  helios.run_experiment_omni_dynamic_time_scales(
    data.toStream, tt_partition, resample = re)(
    summary_dir, 150000, tmpdir)

}
