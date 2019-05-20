import _root_.io.github.mandar2812.dynaml.tensorflow.data._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag

import _root_.org.platanios.tensorflow.api._

import $exec.helios.scripts.csss
import $exec.helios.scripts.env

val exp_dirs =
  (2 to 4).flatMap(i => ls ! home / "tmp" / s"results_exp$i" |? (_.isDir))

val extract_state = (p: Path) => {
  val lines = read.lines ! p / "state.csv"
  lines.head.split(",").zip(lines.last.split(",").map(_.toDouble)).toMap
}

val params_enc = Encoder(
  identityPipe[Map[String, Double]],
  identityPipe[Map[String, Double]]
)

val stabilities = exp_dirs.map(exp_dir => {

  val state = extract_state(exp_dir)

  val triple = timelag.utils.read_cdt_model_preds(
    exp_dir / "test_predictions.csv",
    exp_dir / "test_probabilities.csv",
    exp_dir / "test_data_targets.csv"
  )

  timelag.utils.compute_stability_metrics(
    triple._1,
    triple._2,
    triple._3,
    state,
    params_enc
  )

})

val stabilities2 = exp_dirs2.map(exp_dir => {

  val state = extract_state(exp_dir)

  val triple = timelag.utils.read_cdt_model_preds(
    exp_dir / "test_predictions.csv",
    exp_dir / "test_probabilities.csv",
    exp_dir / "test_data_targets.csv"
  )

  timelag.utils.compute_stability_metrics(
    triple._1,
    triple._2,
    triple._3,
    state,
    helios.learn.cdt_loss.params_enc
  )

})

val fte_exp = csss.experiments()

val fte_stability = fte_exp.map(exp_dir => {

  val state = extract_state(exp_dir)

  val preds =
    (ls ! exp_dir |? (_.segments.last.contains("predictions_test")))(0)
  val probs =
    (ls ! exp_dir |? (_.segments.last.contains("probabilities_test")))(0)

  require(
    preds.segments.last.split('.').head.split('_').last == probs.segments.last
      .split('.')
      .head
      .split('_')
      .last
  )

  val fte_data = (ls ! exp_dir |? (_.segments.last.contains("test_data"))).last

  val triple = fte.data.fte_model_preds(preds, probs, fte_data)

  timelag.utils.compute_stability_metrics(
    triple._1,
    triple._2,
    triple._3,
    state,
    params_enc
  )
})
