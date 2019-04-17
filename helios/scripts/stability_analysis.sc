import _root_.io.github.mandar2812.dynaml.tensorflow.data._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.core.timelag

import _root_.org.platanios.tensorflow.api._

import $exec.helios.scripts.csss
import $exec.helios.scripts.env

val exp_dirs = Seq(
  home / "Downloads" / "results_exp1" / "const_lag_timelag_mo_2019-03-01-20-17-05",
  home / "Downloads" / "results_exp2" / "const_v_timelag_mo_2019-03-01-20-17-40",
  home / "results_exp3" / "const_a_timelag_mo_2019-04-01-21-52-36",
  home / "results_exp4" / "softplus_timelag_mo_2019-04-03-17-50-54",
  //home / "tmp" /  "const_v_timelag_mo_2019-04-08-10-51-12",
  home / "tmp" / "const_v_timelag_mo_2019-04-15-18-48-10"
)

val exp_dirs2 = Seq(
  env.summary_dir / "const_lag_timelag_mo_2019-03-01-20-17-05",
  env.summary_dir / "const_v_timelag_mo_2019-03-01-20-17-40",
  env.summary_dir / "const_a_timelag_mo_2019-04-01-21-52-36",
  env.summary_dir / "softplus_timelag_mo_2019-04-03-17-50-54",
  env.summary_dir / "const_v_timelag_mo_2019-04-08-10-51-12"
)

val extract_state = (p: Path) => {
  val lines = read.lines! p / "state.csv"
  lines.head.split(",").zip(lines.last.split(",").map(_.toDouble)).toMap
}

val exp_dirs = List(
  root/'ufs/'chandork/'tmp/"const_v_timelag_mo_2019-04-16-22-38-58",
  root/'ufs/'chandork/'tmp/"const_a_timelag_mo_2019-04-17-07-00-13",
  root/'ufs/'chandork/'tmp/"softplus_timelag_mo_2019-04-16-23-29-07"
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
    helios.learn.cdt_loss.params_enc)

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
    helios.learn.cdt_loss.params_enc)

})

val fte_exp = csss.experiments() |? (_.segments.last
  .contains("fte_omni_mo_tl_2019-04-05-02-21"))

val fte_stability = fte_exp.map(exp_dir => {

  val state = extract_state(exp_dir)

  val preds =
    (ls ! exp_dir |? (_.segments.last.contains("predictions_test")))(1)
  val probs =
    (ls ! exp_dir |? (_.segments.last.contains("probabilities_test")))(1)

  require(
    preds.segments.last.split('.').head.split('_').last == probs.segments.last
      .split('.')
      .head
      .split('_')
      .last
  )

  val fte_data = (ls ! exp_dir |? (_.segments.last.contains("test_data"))).last

  val triple = fte_model_preds(preds, probs, fte_data)

  compute_stability_metrics(
    triple._1, 
    triple._2, 
    triple._3,
    state, 
    helios.learn.cdt_loss.params_enc)
})
