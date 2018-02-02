import _root_.io.github.mandar2812.PlasmaML.helios
import ammonite.ops._
import org.joda.time._


//Data with MDI images
val data = helios.generate_data_goes()

val test_start = new DateTime(2003, 10, 10, 0, 0)

val test_end   = new DateTime(2003, 10, 31, 23, 59)

val tt_partition = (p: (DateTime, (Path, (Double, Double)))) =>
  if(p._1.isAfter(test_start) && p._1.isBefore(test_end)) false
  else true

val results = helios.run_experiment_goes(data, tt_partition)("extreme_exp_1", 120000)
