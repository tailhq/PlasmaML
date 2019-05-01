import $exec.env
import ammonite.ops.ImplicitWd._

def experiments() =
  ls ! env.summary_dir |? (_.isDir) |? (_.segments.toSeq.last.contains("fte"))


def scatter_plots_test(summary_dir: Path) =
  ls ! summary_dir |? (_.segments.toSeq.last.contains("scatter_test"))

def scatter_plots_train(summary_dir: Path) =
  ls ! summary_dir |? (_.segments.toSeq.last.contains("scatter_train"))

val script = pwd / 'helios / 'scripts / "visualise_tl.R"
