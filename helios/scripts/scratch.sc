import ammonite.ops._
import ammonite.ops.ImplicitWd._
import breeze.numerics._
import breeze.linalg._
import _root_.io.github.mandar2812.dynaml.{utils => dutils}
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import org.joda.time._

val script1 = pwd / 'helios / 'scripts / "visualise_tl_results.R"
val script2 = pwd / 'helios / 'scripts / "visualise_tl_preds.R"

val search_dir = ls ! home / 'Downloads |? (_.segments.last
  .contains("results_exp"))

val relevant_dirs = search_dir || ((p: Path) => ls ! p |? (_.isDir)) // | (_.head)

relevant_dirs.foreach(dir => {

  print("Processing directory ")
  pprint.pprintln(dir)

  val pre =
    if (dir.segments.last.contains("const_v")) "exp2_"
    else if (dir.segments.last.contains("const_a")) "exp3_"
    else if (dir.segments.last.contains("softplus")) "exp4_"
    else ""

  try {
    %%('Rscript, script1, dir, pre)
  } catch {
    case e: Exception => e.printStackTrace()
  }

  try {
    %%('Rscript, script2, dir, pre)
  } catch {
    case e: Exception => e.printStackTrace()
  }

})

def kl(
  gs1: (DenseVector[Double], DenseMatrix[Double]),
  gs2: (DenseVector[Double], DenseMatrix[Double])
): Double = {

  val (m1, c1) = gs1
  val (m2, c2) = gs2

  val d = m2 - m1

  0.5 * (trace(c1 \ c2) + d.t * (c2 \ d) - m1.size + log(det(c2) / det(c1)))

}

val fte_data = {
  fte.data.load_fte_data_bdv(
    home / "Downloads" / "fte",
    fte.data.carrington_rotations,
    true,
    new DateTime(2008, 1, 1, 0, 0, 0),
    new DateTime(2016, 12, 31, 23, 59, 59)
  )(
    0,
    0,
    90d,
    false
  )
}

val max_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => g.max).toSeq
)

val median_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => dutils.median(g.toStream)).toSeq
)

val vsw = {
  fte.data.load_solar_wind_data_bdv(
    new DateTime(2008, 1, 1, 0, 0, 0),
    new DateTime(2016, 12, 31, 23, 59, 59)
  )((48, 72), false, 24, identityPipe[Seq[Double]])
}

type P = (DateTime, (DenseVector[Double], DenseVector[Double]))

val proc_fte_vsw = {
  fte_data
    .join(vsw)
    .filter(DataPipe((p: P) => p._2._1.toArray.toSeq.forall(x => !x.isNaN)))
    .map(
      DataPipe((p: P) => (p._1, (p._2._1(2 to -1), p._2._2)))
    )
    .partition(DataPipe((p: P) => p._1.getYear() != 2015))

}

val (gs_fte_train, gs_fte_test) = (
  dutils.getStats(proc_fte_vsw.training_dataset.data.map(_._2._1)),
  dutils.getStats(proc_fte_vsw.test_dataset.data.map(_._2._1))
)

val (gs_vsw_train, gs_vsw_test) = (
  dutils.getStats(proc_fte_vsw.training_dataset.data.map(_._2._2)),
  dutils.getStats(proc_fte_vsw.test_dataset.data.map(_._2._2))
)

val high_speed_regions = proc_fte_vsw.copy(
  proc_fte_vsw.training_dataset
    .filterNot(DataPipe((p: P) => p._2._2.toArray.toSeq.forall(_ < 750))),
  proc_fte_vsw.test_dataset.filterNot(
    DataPipe((p: P) => p._2._2.toArray.toSeq.forall(_ < 750))
  )
)

def plot_pattern(pattern: P): Unit = {
  spline(fte.data.latitude_grid.zip(pattern._2._1(0 until 180).toArray.toSeq))
  hold()
  spline(fte.data.latitude_grid.zip(pattern._2._1(180 to -1).toArray.toSeq))
  xAxis("latitude (degrees)")
  yAxis("State")
  title("State profiles for " + pattern._1.toString("HH:00 d MMMM, yyyy"))
  legend(Seq("Log FTE", "Br Source Surface"))
  unhold()


  spline(pattern._2._2.toArray.toSeq)
  xAxis(
    "Hours after " + pattern._1.plusHours(48).toString("HH:00 d MMMM, yyyy")
  )
  yAxis("Solar Wind Speed")
  title(
    "Solar Wind profile " + pattern._1
      .plusHours(48)
      .toString("HH:00 d MMMM, yyyy") + " to " + pattern._1
      .plusHours(48 + 72)
      .toString("HH:00 d MMMM, yyyy")
  )
}


val test_labels = {
  val test_files = folds_dtlr.flatMap(p => ls! p |? (_.segments.last.contains("test_data")))

  
}