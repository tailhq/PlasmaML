import breeze.linalg._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RadialDiffusion


val (nL,nT) = (100, 50)
val lMax = 20
val tMax = 20

val lShellLimits = (1.0, 2.0)
val timeLimits = (0.0, 2.0)

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val theta = 0.02
val alpha = 0.005 + theta*math.pow(omega*lShellLimits._2, 2.0)

val referenceSolution = (l: Double, t: Double) => math.sin(omega*l)*(math.exp(-alpha*t) + 1.0)

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

val lShellVec = DenseVector.tabulate[Double](nL+1)(i =>
  if(i < nL) lShellLimits._1+(rds.deltaL*i)
  else lShellLimits._2).toArray.toSeq

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => referenceSolution(l - lShellLimits._1, 0.0)).toArray)

val timeVec = DenseVector.tabulate[Double](nT+1)(i =>
  if(i < nT) timeLimits._1+(rds.deltaT*i)
  else timeLimits._2).toArray.toSeq

val diffProVec = lShellVec.map(l => theta*l*l)
val lossProVec = lShellVec.map(l => alpha - math.pow(l*omega, 2.0)*theta)

val diffProfileGT = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,_) => diffProVec(i))
val lossProfileGT = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => lossProVec(i)/(1 + math.exp(alpha*timeVec(j))))
val boundFluxGT = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => if(i == nL || i == 0) referenceSolution(i * rds.deltaL, j * rds.deltaT) else 0.0)

val radialDiffusionStack = rds.getComputationStack(diffProfileGT, lossProfileGT, boundFluxGT)

val solution = radialDiffusionStack forwardPropagate initialPSDGT

spline(timeVec.zip(solution.map(_(0))))
hold()

(1 to lMax).foreach(l => {
  spline(timeVec.zip(solution.map(_(l))))
})

unhold()

legend(DenseVector.tabulate[Double](lMax+1)(i =>
  if(i < nL) lShellLimits._1+(rds.deltaL*i)
  else lShellLimits._2).toArray.map(s => "L = "+"%3f".format(s)))
title("Evolution of Phase Space Density f(L,t)")
xAxis("time")
yAxis("f(L,t)")


spline(lShellVec.toArray.toSeq.zip(solution.head.toArray.toSeq))
hold()

(1 to tMax).foreach(l => {
  spline(lShellVec.toArray.toSeq.zip(solution(l).toArray.toSeq))
})

unhold()

legend(DenseVector.tabulate[Double](tMax+1)(i =>
  if(i < nL) timeLimits._1+(rds.deltaT*i)
  else timeLimits._2).toArray.map(s => "t = "+"%3f".format(s)))

title("Variation of Phase Space Density f(L,t)")
xAxis("L")
yAxis("f(L,t)")
