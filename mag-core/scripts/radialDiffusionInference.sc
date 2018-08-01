import breeze.linalg._
import breeze.stats.distributions.Gamma
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder, MetaPipe}
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.utils.ConfigEncoding
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._

val (nL,nT) = (200, 50)

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

val (lShellVec, timeVec) = rds.stencil

val Kp = DataPipe((t: Double) =>
  if(t<= 0d) 2.5
  else if(t < 1.5) 2.5 + 4*t
  else if (t >= 1.5 && t< 3d) 8.5
  else if(t >= 3d && t<= 5d) 17.5 - 3*t
  else 2.5)


val baseNoiseLevel = 1.2
val mult = 0.8

/*
* Define parameters of radial diffusion system:
*
*  1) The diffusion field: dll
*  2) The particle injection process: q
*  3) The loss parameter: lambda
*
* Using the MagnetosphericProcessTrend class,
* we define each unknown process using a canonical
* parameterization of diffusion processes in the
* magnetosphere.
*
* For each process we must specify 4 parameters
* alpha, beta, a, b
* */

//Diffusion Field
val (dll_alpha, dll_beta, dll_gamma, dll_b)  = dll_params
val dll_trend                                = MagTrend(Kp, "dll")

val dll_prior = new DiffusionPrior(
  dll_trend,
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new SECovFunc(rds.deltaT*mult, baseNoiseLevel),
  baseNoiseLevel*mult, (dll_alpha, dll_beta, dll_gamma, dll_b))

//Injection process
val (q_alpha, q_beta, q_gamma, q_b) = (0d, 2.5d, 0d, 0.05)
val q_trend                         = MagTrend(Kp, "Q")

val q_prior = new DiffusionPrior(
  q_trend,
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new SECovFunc(rds.deltaT*mult, baseNoiseLevel),
  baseNoiseLevel*mult, (q_alpha, q_beta, q_gamma, q_b))

//Loss Process
val (loss_alpha, loss_beta, loss_gamma, loss_b) = lambda_params
val loss_trend                                  = MagTrend(Kp, "lambda")

val loss_prior = new DiffusionPrior(
  loss_trend,
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new SECovFunc(rds.deltaT*mult, baseNoiseLevel),
  baseNoiseLevel*mult, (loss_alpha, loss_beta, loss_gamma, loss_b))

//Create ground truth diffusion parameter functions
val dll = (l: Double, t: Double) => dll_trend(
  Map(
    "dll_alpha"    -> dll_alpha,
    "dll_beta"     -> dll_beta,
    "dll_gamma"    -> dll_gamma,
    "dll_b"        -> dll_b))((l, t))

val q = (l: Double, t: Double) => q_trend(
  Map(
    "Q_alpha"      -> q_alpha,
    "Q_beta"       -> q_beta,
    "Q_gamma"      -> q_gamma,
    "Q_b"          -> q_b))((l, t))

val lambda = (l: Double, t: Double) => loss_trend(
  Map(
    "lambda_alpha" -> 1/(0.5*1.2*math.pow(10, 4)),
    "lambda_beta"  -> 1d,
    "lambda_gamma" -> loss_gamma,
    "lambda_b"     -> 0.18))((l, t))

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val initialPSD = (l: Double) => math.sin(omega*(l - lShellLimits._1))

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => initialPSD(l)).toArray)

//Create ground truth PSD data and corrupt it with statistical noise.
val groundTruth = rds.solve(q, dll, lambda)(initialPSD)
val ground_truth_matrix = DenseMatrix.horzcat(groundTruth.tail.map(_.asDenseMatrix.t):_*)
val measurement_noise = GaussianRV(0.0, 0.15)

val noise_mat = DenseMatrix.tabulate[Double](nL+1, nT)((_, _) => measurement_noise.draw)
val data: DenseMatrix[Double] = ground_truth_matrix + noise_mat

val covL = new SECovFunc(rds.deltaL*mult, baseNoiseLevel)
covL.block_all_hyper_parameters
val covT = new SECovFunc(rds.deltaT*mult, baseNoiseLevel)
covT.block_all_hyper_parameters

val radialDiffusionProcess = StochasticRadialDiffusion(
  covL, covT,
  q_prior, dll_prior,
  loss_prior)

val blocked_hyper_params = {
  dll_prior.trendParamsEncoder(dll_prior._meanFuncParams).keys ++
    q_prior.trendParamsEncoder(q_prior._meanFuncParams).keys
}


radialDiffusionProcess.block_++(blocked_hyper_params.toSeq:_*)



val hyper_params = radialDiffusionProcess.effective_hyper_parameters

val hyper_prior = getPriorMapDistr(hyper_params.map(h => (h, Gamma(1.0, 1.0))).toMap)
val mapEncoding = ConfigEncoding(hyper_params)

val processed_prior = EncodedContDistrRV(hyper_prior, mapEncoding)

val forward_model = radialDiffusionProcess.forwardModel(lShellLimits, nL, timeLimits, nT) _

val likelihood = DataPipe((hyp: DenseVector[Double]) => {
  val config = mapEncoding.i(hyp)
  radialDiffusionProcess.setState(config)
  radialDiffusionProcess.forwardModel(lShellLimits, nL, timeLimits, nT)(initialPSDGT)
})

implicit val ev = VectorField(hyper_params.length)

val proposal_distr1 = MultGaussianRV(hyper_params.length)(
  DenseVector.zeros[Double](hyper_params.length),
  DenseMatrix.eye[Double](hyper_params.length)*0.2)

val proposal_distr2 = MultStudentsTRV(hyper_params.length)(
  2.5, DenseVector.zeros[Double](hyper_params.length),
  DenseMatrix.eye[Double](hyper_params.length)*0.8)


val mcmc_sampler = new GenericContinuousMCMC[
  DenseVector[Double], DenseMatrix[Double]](
  processed_prior, likelihood, proposal_distr1,
  burnIn = 400, dropCount = 0
)

val post = mcmc_sampler.posterior(data).iid(1000)

val processed_samples: Seq[Map[String, Double]] = post.draw.map(mapEncoding.i(_))

val alphaBeta = processed_samples.map(m => (m("lambda_alpha"), m("lambda_beta")))
val alpha_b = processed_samples.map(m => (m("lambda_alpha"), m("lambda_b")))

scatter(alphaBeta)
xAxis("Loss alpha")
yAxis("Loss beta")
title("Samples from Posterior P(lambda_alpha, lambda_beta | PSD data)")


scatter(alpha_b)
xAxis("Loss Parameter a")
yAxis("Loss Parameter b")
title("Samples from Posterior P(lambda_a, lambda_b | PSD data)")
