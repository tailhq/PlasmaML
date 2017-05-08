import breeze.linalg._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.{RadialDiffusion, StochasticRadialDiffusion}
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.pipes.{Encoder, MetaPipe}
import io.github.mandar2812.dynaml.analysis.implicits._


val (nL,nT) = (500, 50)
val lMax = 20
val tMax = 5

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

val solution = rds.solve(q, dll, boundFlux)(initialPSD)

spline(timeVec.zip(solution.map(_(0))))
hold()

(1 to lMax).foreach(l => {
  spline(timeVec.zip(solution.map(_(l*5))))
})

unhold()

legend(DenseVector.tabulate[Double](lMax+1)(i =>
  if(i < nL) lShellLimits._1+(rds.deltaL*i*5)
  else lShellLimits._2).toArray.map(s => "L = "+"%3f".format(s)))
title("Evolution of Phase Space Density f(L,t)")
xAxis("time")
yAxis("f(L,t)")


spline(lShellVec.toArray.toSeq.zip(solution.head.toArray.toSeq))
hold()

(1 to tMax).foreach(l => {
  spline(lShellVec.toArray.toSeq.zip(solution(l*10).toArray.toSeq))
})

unhold()

legend(DenseVector.tabulate[Double](tMax+1)(i =>
  if(i < nL) timeLimits._1+(rds.deltaT*i*10)
  else timeLimits._2).toArray.map(s => "t = "+"%3f".format(s)))

title("Variation of Phase Space Density f(L,t)")
xAxis("L")
yAxis("f(L,t)")



spline(lShellVec.toArray.map(lShell => (lShell, referenceSolution(lShell, 0.0))).toSeq)
hold()

(1 to tMax).foreach(l => {
  spline(lShellVec.toArray.map(lShell => (lShell, referenceSolution(lShell, timeVec(l*10)))).toSeq)
})

unhold()

legend(DenseVector.tabulate[Double](tMax+1)(i =>
  if(i < nL) timeLimits._1+(rds.deltaT*i*10)
  else timeLimits._2).toArray.map(s => "t = "+"%3f".format(s)))
xAxis("L")
yAxis("f(L,t)")
title("Reference Solution")



val encoder = Encoder(
  (conf: Map[String, Double]) => (conf("c"), conf("s")),
  (cs: (Double, Double)) => Map("c" -> cs._1, "s" -> cs._2))


val dll_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  GaussianSpectralKernel[Double](0.0, 0.2, encoder) + new MAKernel(0.02),
  GaussianSpectralKernel[Double](0.0, 0.2, encoder) + new MAKernel(0.02),
  new MAKernel(0.02), new MAKernel(0.02))(
  MetaPipe((alphaBeta: (Double, Double)) => (x: (Double, Double)) => alphaBeta._1*math.pow(x._1, alphaBeta._2)),
  (0.5, 1.27))

val q_prior = CoRegGPPrior[Double, Double, (Double, Double)](
  new SECovFunc(0.1, 0.1) + new MAKernel(0.02),
  new SECovFunc(0.1, 0.1) + new MAKernel(0.02),
  new MAKernel(0.02), new MAKernel(0.02))(
  MetaPipe((alphaBeta: (Double, Double)) => (x: (Double, Double)) => {
    alphaBeta._1*math.pow(x._1, alphaBeta._2)
  }),
  (0.5, 1.27))

val radialDiffusionProcess = StochasticRadialDiffusion(
  GaussianSpectralKernel[Double](0.0, 1.0, encoder) + new MAKernel(0.01),
  new GenericMaternKernel[Double](0.1, 1) + new MAKernel(0.01),
  q_prior, dll_prior)


val result = radialDiffusionProcess.priorDistribution(lShellLimits, nL, timeLimits, nT)(initialPSDGT)

spline(lShellVec.toArray.map(lShell => (lShell, referenceSolution(lShell, timeLimits._2))).toSeq)

hold()
val samples = (1 to 10).map(_ => {
  val sample_solution = result.draw
  val psd_profile_nt = sample_solution(::, nT-1)
  spline(lShellVec.toArray.toSeq.zip(psd_profile_nt.toArray.toSeq))
})

spline(lShellVec.toArray.toSeq.zip(result.m(::,nT-1).toArray.toSeq))

unhold()
legend(Array("Reference Solution")++(1 to 10).map(i => "Sample: "+i)++Array("Mean"))
xAxis("L")
yAxis("f(L,t)")
title("Generated Samples vs Reference Solution")
