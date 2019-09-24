import $exec.helios.scripts.env
import $exec.helios.scripts.csss
import _root_.io.github.mandar2812.dynaml.evaluation._
import _root_.io.github.mandar2812.dynaml.tensorflow.data._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import org.joda.time._

import _root_.org.platanios.tensorflow.api._

val cv_experiment_name = "csss_exp_fte_cv"

def get_exp_dir(p: Path) =
  p.segments.toSeq.filter(_.contains("fte_omni_mo_tl")).head

def get_bs_exp_dir(p: Path) =
  p.segments.toSeq.filter(_.startsWith("bs_fte_omni_mo_tl")).head

val relevant_files = ls.rec ! env.summary_dir / cv_experiment_name |? (
  p =>
    p.segments.toSeq.last.contains("test") ||
      p.segments.toSeq.last.contains("state.csv") || 
      p.segments.toSeq.last.contains("config.json")
  )

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
  cv_experiment_name
)

// Part to run locally
//Download compressed archive having cv results
%%(
  'scp,
  s"chandork@juniper.md.cwi.nl:~/tmp/${cv_experiment_name}.tar.gz",
  env.summary_dir
)

%%(
  'tar,
  "zxvf",
  env.summary_dir / s"${cv_experiment_name}.tar.gz",
  "-C",
  env.summary_dir
)

//Get local experiment dir, after decompressing archive
val local_exp_dir = env.summary_dir / cv_experiment_name

//Get individual experiment dirs corresponding to each fold
val exps = {
  ls ! local_exp_dir |? (_.segments.toSeq.last.startsWith("fte_omni"))
}

//Construct over all scatter file
val scatter =
  exps
    .map(dir => {
      val sc_files = csss.scatter_plots_test(dir)
      if (sc_files.length > 0) Some(sc_files.last)
      else None
    })
    .filter(_.isDefined)
    .map(_.get)
    .flatMap(read.lines !)

val scatter_file = local_exp_dir / "scatter_test.csv"

os.write.over(scatter_file, scatter.mkString("\n"))

val metrics = new RegressionMetrics(
  scatter
    .map(_.split(',').take(2).map(_.toDouble))
    .toList
    .map(p => (p.head, p.last)),
  scatter.length
)

val bs_exps = {
  ls ! local_exp_dir |? (_.segments.toSeq.last.startsWith("bs_fte_omni"))
}

val bs_scatter =
  bs_exps.map(dir => csss.scatter_plots_test(dir).last).flatMap(read.lines !)

val bs_scatter_file = local_exp_dir / "bs_scatter_test.csv"

os.write.over(bs_scatter_file, bs_scatter.mkString("\n"))

val bs_metrics = new RegressionMetrics(
  bs_scatter
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

//Generate time series reconstruction for the first fold in cv
val first_exp_scatter = read.lines ! csss
  .scatter_plots_test(exps.head)
  .last | (_.split(',').map(_.toDouble))

val (ts_pred, ts_actual) = first_exp_scatter.zipWithIndex
  .map(
    ti =>
      ((ti._2 + ti._1.last * 6, ti._1.head), (ti._2 + ti._1.last * 6, ti._1(1)))
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

val pred_df = dtfdata.dataset(pred).to_zip(identityPipe[(Double, Double)])

val actual_df =
  dtfdata.dataset(actual).to_zip(identityPipe[(Double, Double)])

val ts_df = pred_df join actual_df

val metrics_ts = new RegressionMetrics(
  ts_df.map(tup2_2[Double, (Double, Double)]).data.toList,
  ts_df.size
)

os.write.over(
  local_exp_dir / "rec_ts_actual.csv",
  actual.map(x => s"${x._1},${x._2}").mkString("\n")
)
os.write.over(
  local_exp_dir / "rec_ts_predicted.csv",
  pred.map(x => s"${x._1},${x._2}").mkString("\n")
)

val process_ts_preds = IterableFlatMapPipe(DataPipe((exp: Path) => {
  val exp_scatter = read.lines ! csss
    .scatter_plots_test(exp)
    .last | (_.split(',').map(_.toDouble))

  val (ts_pred, ts_actual) = exp_scatter.zipWithIndex
    .map(
      ti =>
        ((ti._2 + ti._1.last * 6, ti._1.head), (ti._2 + ti._1.last, ti._1(1)))
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

  val pred_df = dtfdata.dataset(pred).to_zip(identityPipe[(Double, Double)])

  val actual_df =
    dtfdata.dataset(actual).to_zip(identityPipe[(Double, Double)])

  val ts_df = pred_df join actual_df

  ts_df.map(tup2_2[Double, (Double, Double)]).data
}))

val scatter_ts_cv = process_ts_preds(exps).toList

val metrics_ts_cv = new RegressionMetrics(
  scatter_ts_cv.toList,
  scatter_ts_cv.length
)

val params_enc = Encoder(
  identityPipe[Map[String, Double]],
  identityPipe[Map[String, Double]]
)

val extract_state = (p: Path) => {
  val lines = read.lines ! p / "state.csv"
  lines.head.split(",").zip(lines.last.split(",").map(_.toDouble)).toMap
}

val stabilities = exps.map(exp_dir => {

  val state = extract_state(exp_dir)

  val preds = csss.test_data_preds(exp_dir).head
  val probs = csss.test_data_probs(exp_dir).head

  require(
    preds.segments.toSeq.last
      .split('.')
      .head
      .split('_')
      .last == probs.segments.toSeq.last
      .split('.')
      .head
      .split('_')
      .last
  )

  val fte_data = csss.test_data(exp_dir).last

  val triple = fte.data.fte_model_preds(preds, probs, fte_data)

  timelag.utils.compute_stability_metrics(
    triple._1,
    triple._2,
    triple._3,
    state,
    params_enc
  )
})

val metrics_rtl = {
  val pt = exps.flatMap(exp => {

    val preds_file = csss.test_preds(exp).last

    val preds = read.lines ! preds_file | (
      line => line.split(",").map(_.toDouble)
    )

    val test_data_file = csss.test_data(exp).last

    val test_data =
      fte.data
        .read_json_data_file(
          test_data_file,
          identityPipe[Array[Double]],
          identityPipe[Array[Double]]
        )
        .map(
          tup2_2[DateTime, (Array[Double], Array[Double])] > tup2_2[Array[
            Double
          ], Array[Double]]
        )
        .data

    preds.zip(test_data).flatMap(c => c._1.zip(c._2).toSeq).toSeq

  })

  new RegressionMetrics(pt.toList, pt.length)
}


/* val metrics_readjusted = {
  exps.map(exp => {

  })
} */