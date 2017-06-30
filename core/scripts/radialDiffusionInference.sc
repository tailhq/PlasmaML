import breeze.linalg._
import breeze.stats.distributions.Gamma
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.{
DiffusionParameterTrend, DiffusionPrior, RadialDiffusion, StochasticRadialDiffusion}
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder, MetaPipe}
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.utils.ConfigEncoding


val (nL,nT) = (200, 50)

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

val Kp = DataPipe((t: Double) =>
  if(t<= 0d) 2.5
  else if(t < 1.5) 2.5 + 4*t
  else if (t >= 1.5 && t< 3d) 8.5
  else if(t >= 3d && t<= 5d) 17.5 - 3*t
  else 2.5)


val baseNoiseLevel = 1.2
val mult = 0.8


//Define parameters of radial diffusion system
val dll_alpha = 1d
val dll_beta = 10d
val dll_a = -9.325
val dll_b = 0.506

val dll_trend = new DiffusionParameterTrend[Map[String, Double]](Kp)(DiffusionParameterTrend.getEncoder("dll"))
val dll_prior = new DiffusionPrior(
  dll_trend,
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new SECovFunc(rds.deltaT*mult, baseNoiseLevel),
  baseNoiseLevel*mult, (dll_alpha, dll_beta, dll_a, dll_b))

val q_alpha = 1d
val q_beta = 0d
val q_a = 0.002d
val q_b = 0.05d

val q_trend = new DiffusionParameterTrend[Map[String, Double]](Kp)(DiffusionParameterTrend.getEncoder("Q"))
val q_prior = new DiffusionPrior(
  q_trend,
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new SECovFunc(rds.deltaT*mult, baseNoiseLevel),
  baseNoiseLevel*mult, (q_alpha, q_beta, q_a, q_b))


val loss_alpha = 1d
val loss_beta = 10d
val loss_a = -9.325
val loss_b = 0.506

val loss_trend = new DiffusionParameterTrend[Map[String, Double]](Kp)(DiffusionParameterTrend.getEncoder("lambda"))
val loss_prior = new DiffusionPrior(
  loss_trend,
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new SECovFunc(rds.deltaT*mult, baseNoiseLevel),
  baseNoiseLevel*mult, (loss_alpha, loss_beta, loss_a, loss_b))


//Create ground truth diffusion parameter functions

val dll = (l: Double, t: Double) => dll_trend(
  Map("dll_alpha" -> dll_alpha, "dll_beta" -> dll_beta, "dll_a" -> dll_a, "dll_b" -> dll_b))(
  (l, t)
)

val q = (l: Double, t: Double) => q_trend(
  Map("Q_alpha" -> q_alpha, "Q_beta" -> q_beta, "Q_a" -> q_a, "Q_b" -> q_b))(
  (l, t)
)

val lambda = (l: Double, t: Double) => loss_trend(
  Map("lambda_alpha" -> loss_alpha, "lambda_beta" -> loss_beta, "lambda_a" -> loss_a, "lambda_b" -> loss_b))(
  (l, t)
)

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val initialPSD = (l: Double) => math.sin(omega*(l - lShellLimits._1))

val groundTruth = rds.solve(q, dll, lambda)(initialPSD)

val ground_truth_matrix = DenseMatrix.horzcat(groundTruth.map(_.asDenseMatrix.t):_*)
val measurement_noise = GaussianRV(0.0, 0.25)
val data = ground_truth_matrix.map(v => v + measurement_noise.draw)