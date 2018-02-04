import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import org.joda.time._


//Data with MDI images
val data = helios.generate_data_goes()

val test_year = 2001

val flux_threshold = -6.5d

val test_start = new DateTime(test_year, 1, 1, 0, 0)

val test_end   = new DateTime(test_year, 12, 31, 23, 59)

val tt_partition = (p: (DateTime, (Path, (Double, Double)))) =>
  if(p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2._1 >= flux_threshold) false
  else true

val results = helios.run_experiment_goes(data, tt_partition)("extreme_exp_"+test_year, 120000)
