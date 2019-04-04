import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import $exec.env
import ammonite.ops.ImplicitWd._

def experiments() =
  ls ! env.summary_dir |? (_.isDir) |? (_.segments.last.contains("fte"))


def scatter_plots_test(csss_exp: helios.Experiment[Double, fte.ModelRunTuning, fte.data.FteOmniConfig]) =
  ls ! csss_exp.results.summary_dir |? (_.segments.last.contains("scatter_test"))

def scatter_plots_train(csss_exp: helios.Experiment[Double, fte.ModelRunTuning, fte.data.FteOmniConfig]) =
  ls ! csss_exp.results.summary_dir |? (_.segments.last.contains("scatter_train"))

val script = pwd / 'helios / 'scripts / "visualise_tl.R"
