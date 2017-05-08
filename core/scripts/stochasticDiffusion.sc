import breeze.linalg._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.{RadialDiffusion, StochasticRadialDiffusion}
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.pipes.{Encoder, MetaPipe}
import io.github.mandar2812.dynaml.probability.{MatrixNormalRV, MeasurableFunction}


val (nL,nT) = (500, 50)

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

val boundFlux = (l: Double, t: Double) => {
  if(l == lShellLimits._1 || l == lShellLimits._2) referenceSolution(l, t) else 0.0
}

val initialPSD = (l: Double) => referenceSolution(l, 0.0)

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT, false)

val lShellVec = DenseVector.tabulate[Double](nL+1)(i =>
  if(i < nL) lShellLimits._1+(rds.deltaL*i)
  else lShellLimits._2).toArray.toSeq

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => referenceSolution(l, 0.0)).toArray)

val timeVec = DenseVector.tabulate[Double](nT+1)(i =>
  if(i < nT) timeLimits._1+(rds.deltaT*i)
  else timeLimits._2).toArray.toSeq


val baseNoiseLevel = 0.001

val encoder = Encoder(
  (conf: Map[String, Double]) => (conf("c"), conf("s")),
  (cs: (Double, Double)) => Map("c" -> cs._1, "s" -> cs._2))


val dll_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  new SECovFunc(rds.deltaL, 0.001) + new MAKernel(baseNoiseLevel),
  new SECovFunc(rds.deltaT, 0.001) + new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel), new MAKernel(baseNoiseLevel))(
  MetaPipe((alphaBeta: (Double, Double)) => (x: (Double, Double)) => {
    alphaBeta._1*math.pow(x._1, alphaBeta._2)
  }),
  (alpha, beta))

val q_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  new SECovFunc(rds.deltaL, 0.001) + new MAKernel(baseNoiseLevel),
  new SECovFunc(rds.deltaT, 0.001) + new MAKernel(baseNoiseLevel),
  new MAKernel(baseNoiseLevel), new MAKernel(baseNoiseLevel))(
  MetaPipe((alphaBeta: (Double, Double)) => (lt: (Double, Double)) => {
    val (l,t) = lt
    val (alp, bet) = alphaBeta
    b*math.sin(a*(l - lShellLimits._1))*math.exp(b*t) -
      a*alp*(bet-2d)*math.pow(l, bet-1d)*(math.exp(b*t) - 1.0)*math.cos(a*(l - lShellLimits._1)) +
      a*a*alp*math.pow(l, bet)*(math.exp(b*t) - 1.0)*math.sin(a*(l - lShellLimits._1))
  }),
  (alpha, beta))

val radialDiffusionProcess = StochasticRadialDiffusion(
  new SECovFunc(rds.deltaL, rds.deltaL),
  new SECovFunc(rds.deltaT, rds.deltaT),
  q_prior, dll_prior)


val result = radialDiffusionProcess.priorDistribution(lShellLimits, nL, timeLimits, nT)(initialPSDGT)

val referenceSolutionMatrix = DenseMatrix.tabulate[Double](nL+1, nT)((i,j) => {
  referenceSolution(lShellVec(i), timeVec(j+1))
})

val errorFunctional = MeasurableFunction[DenseMatrix[Double], Double, MatrixNormalRV](
  RadialDiffusion.error(referenceSolutionMatrix) _) _

val error = errorFunctional(result)



val thalf = timeVec(nT/2)

spline(lShellVec.toArray.map(lShell => (lShell, referenceSolution(lShell, thalf))).toSeq)

hold()
val samples = (1 to 10).map(_ => {
  val sample_solution = result.draw
  val psd_profile_nt = sample_solution(::, nT/2)
  spline(lShellVec.toArray.toSeq.zip(psd_profile_nt.toArray.toSeq))
})

spline(lShellVec.toArray.toSeq.zip(result.m(::,nT/2).toArray.toSeq))

unhold()
legend(Array("Reference Solution")++(1 to 10).map(i => "Sample: "+i)++Array("Mean"))
xAxis("L")
yAxis("f(L,t)")
title("Generated Samples vs Reference Solution")

val dll_dist = dll_prior.priorDistribution(lShellVec, timeVec)


val num_samples = 20
spline(lShellVec.zip(dll_dist.m(::,nT/2).toArray.toSeq))
hold()
(1 to num_samples).map(_ => {
  val dll_sample = dll_dist.draw

  spline(lShellVec.zip(dll_sample(::,nT/2).toArray.toSeq))

})
unhold()
legend(Array("Mean dLL")++(1 to num_samples).map(i => "Sample: "+i))
xAxis("L")
yAxis("dLL(L,t)")
title("Diffusion Process Prior")


val q_dist = q_prior.priorDistribution(lShellVec, timeVec)


spline(lShellVec.zip(q_dist.m(::,nT/2).toArray.toSeq))
hold()
(1 to num_samples).map(_ => {
  val q_sample = q_dist.draw

  spline(lShellVec.zip(q_sample(::,nT/2).toArray.toSeq))

})
unhold()
legend(Array("Mean Q(L, t)")++(1 to num_samples).map(i => "Sample: "+i))
xAxis("L")
yAxis("Q(L,t)")
title("Injection Process Prior")
