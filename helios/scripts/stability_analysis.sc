import _root_.io.github.mandar2812.dynaml.tensorflow.data._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag
import _root_.org.json4s._
import _root_.org.json4s.JsonDSL._
import _root_.org.json4s.jackson.Serialization.{
  read => read_json,
  write => write_json
}
import org.json4s.jackson.JsonMethods._

import _root_.org.platanios.tensorflow.api._

import $exec.helios.scripts.csss
import $exec.helios.scripts.env

implicit val formats = DefaultFormats + FieldSerializer[Map[String, Any]]()

case class CDTStability(c1: Double, c2: Double, n: Int) {

  val is_stable: Boolean = c2 < 2 * c1
}

type ZipPattern = ((Tensor[Double], Tensor[Double]), Tensor[Double])

type StabTriple = Tensor[Double]

type DataTriple =
  (DataSet[Tensor[Double]], DataSet[Tensor[Double]], DataSet[Tensor[Double]])

def compute_stability_metrics(
  predictions: DataSet[Tensor[Double]],
  probabilities: DataSet[Tensor[Double]],
  targets: DataSet[Tensor[Double]]
): CDTStability = {

  val n = predictions.data.head.shape(0)

  val compute_metrics = DataPipe[ZipPattern, StabTriple](zp => {

    val ((y, p), t) = zp

    val sq_error = y.subtract(t).square

    val s0 = sq_error.mean().scalar

    val c1 = p.multiply(sq_error).sum().scalar

    val c2 = p.multiply(sq_error.subtract(c1).square).sum().scalar

    dtf.tensor_f64(4)(s0, c1, c2, 1.0)
  })

  val result = predictions
    .zip(probabilities)
    .zip(targets)
    .map(compute_metrics)
    .reduce(DataPipe2[StabTriple, StabTriple, StabTriple](_ + _))

  val s0 = result(0).divide(result(3)).scalar

  val c1 = result(1).divide(result(3)).scalar / s0

  val c2 = result(2).divide(result(3)).scalar / (s0 * s0)

  CDTStability(c1, c2, n)
}

def read_cdt_model_preds(
  preds: Path,
  probs: Path,
  targets: Path
): DataTriple = {

  val read_lines = DataPipe[Path, Iterable[String]](read.lines ! _)

  val split_lines = IterableDataPipe(
    (line: String) => line.split(',').map(_.toDouble)
  )

  val load_into_tensor = IterableDataPipe(
    (ls: Array[Double]) => dtf.tensor_f64(ls.length)(ls.toSeq: _*)
  )

  val load_data = read_lines > split_lines > load_into_tensor

  (
    dtfdata.dataset(Seq(preds)).flatMap(load_data),
    dtfdata.dataset(Seq(probs)).flatMap(load_data),
    dtfdata.dataset(Seq(targets)).flatMap(load_data)
  )

}

def fte_model_preds(preds: Path, probs: Path, fte_data: Path): DataTriple = {

  val read_file = DataPipe((p: Path) => read.lines ! p)

  val split_lines = IterableDataPipe(
    (line: String) => line.split(',').map(_.toDouble)
  )

  val load_into_tensor = IterableDataPipe(
    (ls: Array[Double]) => dtf.tensor_f64(ls.length)(ls.toSeq: _*)
  )

  val filter_non_empty_lines = IterableDataPipe((l: String) => !l.isEmpty)

  val read_json_record = IterableDataPipe((s: String) => parse(s))

  val load_targets = IterableDataPipe((record: JValue) => {
    val targets_seq = record
      .findField(p => p._1 == "targets")
      .get
      ._2
      .values
      .asInstanceOf[List[Double]]

    val targets = dtf.tensor_f64(targets_seq.length)(targets_seq: _*)

    targets
  })

  val pipeline_fte = read_file > filter_non_empty_lines > read_json_record > load_targets

  val pipeline_model = read_file > split_lines > load_into_tensor

  (
    dtfdata.dataset(Seq(preds)).flatMap(pipeline_model),
    dtfdata.dataset(Seq(probs)).flatMap(pipeline_model),
    dtfdata.dataset(Seq(fte_data)).flatMap(pipeline_fte)
  )

}

val exp_dirs = Seq(
  home / "Downloads" / "results_exp2" / "const_v_timelag_mo_2019-03-01-20-17-40",
  home / "results_exp3" / "const_a_timelag_mo_2019-04-01-21-52-36",
  home / "results_exp4" / "softplus_timelag_mo_2019-04-03-17-50-54"
)

val stabilities = exp_dirs.map(exp_dir => {
  val triple = read_cdt_model_preds(
    exp_dir / "test_predictions.csv",
    exp_dir / "test_probabilities.csv",
    exp_dir / "test_data_targets.csv"
  )

  compute_stability_metrics(triple._1, triple._2, triple._3)

})

val fte_exp = csss.experiments() |? (_.segments.last.contains("fte_omni_mo_tl_2019-04-04-14-49"))

val fte_stability = fte_exp.map(exp_dir => {
  val preds =
    (ls ! exp_dir |? (_.segments.last.contains("predictions_test"))).last
  val probs =
    (ls ! exp_dir |? (_.segments.last.contains("probabilities_test"))).last

  require(
    preds.segments.last.split('.').head.split('_').last == probs.segments.last
      .split('.')
      .head
      .split('_')
      .last
  )

  val fte_data = (ls ! exp_dir |? (_.segments.last.contains("test_data"))).last

  val triple = fte_model_preds(preds, probs, fte_data)

  compute_stability_metrics(triple._1, triple._2, triple._3)
})
