import $exec.helios.scripts.env
import $exec.helios.scripts.csss

val cv_experiment_name = "exp_cv_gs"

val relevant_files = ls.rec ! env.summary_dir / 'csss_exp_fte_cv |? (
  p => p.segments.toSeq.last.contains("test") /* &&
      !p.segments.toSeq.last.contains("test_data") */
)

def get_exp_dir(p: Path) =
  p.segments.toSeq.filter(_.contains("fte_omni_mo_tl")).head

val files_grouped = relevant_files
  .map(p => (get_exp_dir(p), p))
  .groupBy(_._1)
  .mapValues(_.map(_._2))

val main_dir = home / 'tmp / cv_experiment_name
mkdir ! main_dir

files_grouped.foreach(cv => {
  val sub_exp_dir = main_dir / cv._1
  if (!exists(sub_exp_dir)) mkdir ! sub_exp_dir
  cv._2.foreach(f => {
    pprint.pprintln(s"$f -> ${sub_exp_dir}")
    os.copy.into(f, sub_exp_dir)
  })
  pprint.pprintln(sub_exp_dir)
})

%%(
  'tar,
  "-C",
  home / 'tmp,
  "-zcvf",
  home / 'tmp / s"${cv_experiment_name}.tar.gz",
  "exp_cv_gs"
)

// Part to run locally
//Download compressed archive having cv results
%%('scp, s"chandork@juniper.md.cwi.nl:~/tmp/${cv_experiment_name}.tar.gz", home/'tmp)

//Get local experiment dir, after decompressing archive
val local_exp_dir = home / 'tmp / cv_experiment_name

//Get individual experiment dirs corresponding to each fold
val exps = {
  ls ! local_exp_dir |? (_.segments.toSeq.last.contains("fte_omni"))
}

//Construct over all scatter file
val scatter =
  exps.map(dir => csss.scatter_plots_test(dir).last).flatMap(read.lines !)

val scatter_file = local_exp_dir / "scatter_test.csv"

os.write(scatter_file, scatter)

val metrics = new RegressionMetrics(
  scatter
    .map(_.split(',').take(2).map(_.toDouble))
    .toList
    .map(p => (p.head, p.last)),
  scatter.length
)

try {
  %%(
    'Rscript,
    csss.script,
    local_exp_dir,
    scatter_file,
    "test_"
  )
} catch {
  case e: Exception => e.printStackTrace()
}


//Generate time series reconstruction for last fold in cv
val last_exp_scatter = read.lines ! csss
  .scatter_plots_test(exps.last)
  .last | (_.split(',').map(_.toDouble))

val (ts_pred, ts_actual) = last_exp_scatter.zipWithIndex
  .map(
    ti => ((ti._2 + ti._1.last, ti._1.head), (ti._2 + ti._1.last, ti._1(1)))
  )
  .unzip

val pred = ts_pred
  .groupBy(_._1)
  .mapValues(p => {
    val v = p.map(_._2)
    v.sum / p.length
  })
  .toSeq
  .sortBy(_._1)

val actual = ts_actual
  .groupBy(_._1)
  .mapValues(p => {
    val v = p.map(_._2)
    v.sum / p.length
  })
  .toSeq
  .sortBy(_._1)




line(pred)
hold()
line(actual)
legend("Predicted Speed", "Actual Speed")
xAxis("time")
yAxis("Solar Wind Speed (km/s)")
unhold()
