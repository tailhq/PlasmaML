import breeze.linalg._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RadialDiffusion

val (nL,nT) = (500, 50)

val lMax = 10
val tMax = 5

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val theta = 0.006
val alpha = 0.005 + theta*math.pow(omega*lShellLimits._2, 2.0)

val referenceSolution = (l: Double, t: Double) => math.sin(omega*(l - lShellLimits._1))*math.exp(-alpha*t)

val radialDiffusionSolver = (binsL: Int, binsT: Int) => new RadialDiffusion(lShellLimits, timeLimits, binsL, binsT)

val boundFlux = (l: Double, t: Double) => {
  if(l == lShellLimits._1 || l == lShellLimits._2) referenceSolution(l, t) else 0.0
}

val dll = (l: Double, _: Double) => theta*l*l
val q = (l: Double, _: Double) => alpha - math.pow(l*omega, 2.0)*theta

val initialPSD = (l: Double) => referenceSolution(l, 0.0)


val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT, true)

val (lShellVec, timeVec) = RadialDiffusion.buildStencil(lShellLimits, nL, timeLimits, nT)

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => referenceSolution(l, 0.0)).toArray)


val solution = rds.solve(q, dll, boundFlux)(initialPSD)

/*
*
* First set of plots
*
* */

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


/*
*
* Second set of plots
*
* */
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

