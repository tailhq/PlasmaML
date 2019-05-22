import $exec.env
import ammonite.ops.ImplicitWd._
import _root_.io.github.mandar2812.dynaml.pipes._

def experiments() =
  ls ! env.summary_dir |? (_.isDir) |? (_.segments.last.contains("fte"))


def scatter_plots_test(summary_dir: Path) =
  ls ! summary_dir |? (_.segments.last.contains("scatter_test"))

def scatter_plots_train(summary_dir: Path) =
  ls ! summary_dir |? (_.segments.last.contains("scatter_train"))

val script = pwd / 'helios / 'scripts / "visualise_tl.R"

val time_window = (48, 72)

val avg_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => g.sum / g.length).toSeq
)

val max_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => g.max).toSeq
)

val median_sw_6h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(6).map(g => dutils.median(g.toStream)).toSeq
)


val avg_sw_24h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(24).map(g => g.sum / g.length).toSeq
)

val max_sw_24h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(24).map(g => g.max).toSeq
)

val median_sw_24h = DataPipe(
  (xs: Seq[Double]) => xs.grouped(24).map(g => dutils.median(g.toStream)).toSeq
)

val fact            = 10
val base_iterations = 80000
def ext_iterations  = base_iterations * fact

val base_it_pdt = 3
def ext_it_pdt  = base_it_pdt + 2
