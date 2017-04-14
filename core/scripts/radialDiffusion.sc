import breeze.linalg._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RadialDiffusion


val (nL,nT) = (100, 50)
val lMax = 20
val tMax = 5

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)
val a = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val b = math.log(2d)/timeLimits._2

val referenceSolution = (l: Double, t: Double) => math.sin(a*(l - lShellLimits._1))*(math.exp(b*t) - 1.0)

//Define parameters of radial diffusion system
val alpha = 1.9E-10
val beta = 11.7

val dll = (l: Double, _: Double) => alpha*math.pow(l, beta)

val q = (l: Double, t: Double) => {
  b*math.sin(a*(l - lShellLimits._1))*math.exp(b*t) -
    a*alpha*(beta-2d)*math.pow(l, beta-1d)*(math.exp(b*t) - 1.0)*math.cos(a*(l - lShellLimits._1)) +
    a*a*alpha*math.pow(l, beta)*(math.exp(b*t) - 1.0)*math.cos(a*(l - lShellLimits._1))
}

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT, false)

val lShellVec = DenseVector.tabulate[Double](nL+1)(i =>
  if(i < nL) lShellLimits._1+(rds.deltaL*i)
  else lShellLimits._2).toArray.toSeq

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => referenceSolution(l, 0.0)).toArray)

val timeVec = DenseVector.tabulate[Double](nT+1)(i =>
  if(i < nT) timeLimits._1+(rds.deltaT*i)
  else timeLimits._2).toArray.toSeq

val diffProfileGT = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => dll(lShellVec(i), timeVec(j)))
val injectionProfileGT = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => q(lShellVec(i), timeVec(j)))
val boundFluxGT = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => {
  if(i == nL || i == 0) referenceSolution(lShellVec(i), timeVec(j))
  else 0.0
})

val radialDiffusionStack = rds.getComputationStack(injectionProfileGT, diffProfileGT, boundFluxGT)

val solution = radialDiffusionStack forwardPropagate initialPSDGT

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
