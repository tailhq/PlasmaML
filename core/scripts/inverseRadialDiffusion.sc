import breeze.linalg._
import breeze.stats.distributions.Gamma
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.{RadialDiffusion, StochasticRadialDiffusion}
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder, MetaPipe}
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.PlasmaML.utils._
import io.github.mandar2812.dynaml.analysis.VectorField


val (nL,nT) = (200, 50)

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)
val a = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val b = math.log(2d)/timeLimits._2

val referenceSolution = (l: Double, t: Double) => math.sin(a*(l - lShellLimits._1))*(math.exp(b*t) - 1.0)

//Define parameters of radial diffusion system
val alpha = 0.9
val beta = 2.0

val dll = (l: Double, _: Double) => alpha*math.pow(l, beta)

val q = (l: Double, t: Double) => {
  b*math.sin(a*(l - lShellLimits._1))*math.exp(b*t) -
    a*alpha*(beta-2d)*math.pow(l, beta-1d)*(math.exp(b*t) - 1.0)*math.cos(a*(l - lShellLimits._1)) +
    a*a*alpha*math.pow(l, beta)*(math.exp(b*t) - 1.0)*math.sin(a*(l - lShellLimits._1))
}

val initialPSD = (l: Double) => referenceSolution(l, 0.0)

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

val (lShellVec, timeVec) = rds.stencil

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => referenceSolution(l, 0.0)).toArray)


val baseNoiseLevel = 1.2
val mult = 0.8

val encoder = Encoder(
  (conf: Map[String, Double]) => (conf("c"), conf("s")),
  (cs: (Double, Double)) => Map("c" -> cs._1, "s" -> cs._2))


val dll_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel))(
  MetaPipe((alphaBeta: (Double, Double)) => (x: (Double, Double)) => {
    alphaBeta._1*math.pow(x._1, alphaBeta._2)
  }),
  Encoder(
    (alphaBeta: (Double, Double)) => Map("d_alpha"-> alphaBeta._1, "d_beta"-> alphaBeta._2),
    (conf: Map[String, Double]) => (conf("d_alpha"), conf("d_beta"))
  ),
  (alpha, beta))

val q_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  new SECovFunc(rds.deltaL*mult, baseNoiseLevel),
  new GenericMaternKernel[Double](rds.deltaT*mult, 1),
  new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel))(
  MetaPipe((alphaBeta: (Double, Double)) => (lt: (Double, Double)) => {
    val (l,t) = lt
    val (alp, bet) = alphaBeta
    b*math.sin(a*(l - lShellLimits._1))*math.exp(b*t) -
      a*alp*(bet-2d)*math.pow(l, bet-1d)*(math.exp(b*t) - 1.0)*math.cos(a*(l - lShellLimits._1)) +
      a*a*alp*math.pow(l, bet)*(math.exp(b*t) - 1.0)*math.sin(a*(l - lShellLimits._1))
  }),
  Encoder(
    (alphaBeta: (Double, Double)) => Map("q_alpha"-> alphaBeta._1, "q_beta"-> alphaBeta._2),
    (conf: Map[String, Double]) => (conf("q_alpha"), conf("q_beta"))
  ),
  (2.0, 1.0))


val loss_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel))(
  MetaPipe((_: (Double, Double)) => (_: (Double, Double)) => 0.0),
  Encoder(
    (alphaBeta: (Double, Double)) => Map("l_alpha"-> alphaBeta._1, "l_beta"-> alphaBeta._2),
    (conf: Map[String, Double]) => (conf("l_alpha"), conf("l_beta"))
  ),
  (2.0, 1.0)
)

val covL = new SECovFunc(rds.deltaL*mult, baseNoiseLevel)
covL.block_all_hyper_parameters
val covT = new SECovFunc(rds.deltaT*mult, baseNoiseLevel)
covT.block_all_hyper_parameters

val radialDiffusionProcess = StochasticRadialDiffusion(
  covL, covT,
  q_prior, dll_prior,
  loss_prior)

radialDiffusionProcess.block_++(loss_prior.trendParamsEncoder(loss_prior._meanFuncParams).keys.toSeq:_*)
radialDiffusionProcess.block_++(dll_prior.trendParamsEncoder(loss_prior._meanFuncParams).keys.toSeq:_*)


val hyper_params = radialDiffusionProcess.effective_hyper_parameters

val hyper_prior = getPriorMapDistr(hyper_params.map(h => (h, Gamma(1.0, 1.0))).toMap)
val mapEncoding = ConfigEncoding(hyper_params)

val processed_prior: ContinuousDistrRV[DenseVector[Double]] = EncodedContDistrRV(hyper_prior, mapEncoding)

val forward_model = radialDiffusionProcess.forwardModel(lShellLimits, nL, timeLimits, nT) _

val likelihood = DataPipe((hyp: DenseVector[Double]) => {
  val config = mapEncoding.i(hyp)
  radialDiffusionProcess.setState(config)
  radialDiffusionProcess.forwardModel(lShellLimits, nL, timeLimits, nT)(initialPSDGT)
})

implicit val ev = VectorField(hyper_params.length)

val proposal_distr1 = MultGaussianRV(hyper_params.length)(
  DenseVector.zeros[Double](hyper_params.length),
  DenseMatrix.eye[Double](hyper_params.length))

val proposal_distr2 = MultStudentsTRV(hyper_params.length)(
  3.5, DenseVector.zeros[Double](hyper_params.length),
  DenseMatrix.eye[Double](hyper_params.length)*0.5)


val mcmc_sampler = new ContinuousMCMC[DenseVector[Double], DenseMatrix[Double]](
  processed_prior, likelihood, proposal_distr1,
  burnIn = 0, dropCount = 0
)

val measurement_noise = GaussianRV(0.0, 0.25)

val referenceSolutionMatrix = DenseMatrix.tabulate[Double](nL+1, nT)((i,j) => {
  referenceSolution(lShellVec(i), timeVec(j+1)) + measurement_noise.draw
})

val post = mcmc_sampler.posterior(referenceSolutionMatrix).iid(2000)

val processed_samples: Seq[Map[String, Double]] = post.draw.map(mapEncoding.i(_))

val alphaBeta = processed_samples.map(m => (m("q_alpha"), m("q_beta")))


scatter(alphaBeta)
xAxis("Injection alpha")
yAxis("Injection beta")
title("Samples from Posterior P(alpha, beta | PSD data)")
