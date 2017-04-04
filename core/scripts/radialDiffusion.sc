import breeze.stats.distributions._
import breeze.linalg._
import io.github.mandar2812.PlasmaML.RadialDiffusionSolver
import io.github.mandar2812.dynaml.probability._
import com.quantifind.charts.Highcharts._


val (nL,nT) = (20, 20)
val lMax = 15
val tMax = 15

val lShellLimits = (1.5, 6.5)

val l_center1 = (lShellLimits._1*0.75)+(lShellLimits._2*0.25)
val l_center2 = (lShellLimits._1*0.25)+(lShellLimits._2*0.75)



val timeLimits = (0.0, 10.0)

val initialPSD = DenseVector.tabulate(nL+1)(l =>
  math.exp(-0.05*math.abs(l - l_center1)) + math.exp(-0.05*math.abs(l - l_center2)))

val rds = new RadialDiffusionSolver(lShellLimits, timeLimits, nL, nT)

val lShellVec = DenseVector.tabulate[Double](nL+1)(i =>
  if(i < nL) lShellLimits._1+(rds.deltaL*i)
  else lShellLimits._2).toArray.toSeq

val timeVec = DenseVector.tabulate[Double](nT+1)(i =>
  if(i < nL) timeLimits._1+(rds.deltaT*i)
  else timeLimits._2).toArray.toSeq


val ru = RandomVariable(new Uniform(0.0, 2.0))
val gau = GaussianRV(0.0, 1.5)
val rg = RandomVariable(new Beta(1.5, 4.5))

val randMatrix = (rv: RandomVariable[Double]) => (i: Int, j: Int) => DenseMatrix.tabulate[Double](i, j)((l,m) => rv.draw)

val randMatrixB = (rv: RandomVariable[Double]) => (i: Int, j: Int) => DenseMatrix.tabulate[Double](i, j)((l,m) => if (l == i-1) -rv.draw else if(l == 0) rv.draw else 0.0)

val (diffusionProfile, lossProfile, boundaryFlux) = (
  randMatrix(rg)(nL+1,nT),
  randMatrix(rg)(nL+1,nT),
  randMatrixB(ru)(nL+1,nT))

val radialDiffusionStack = rds.getComputationStack(diffusionProfile, lossProfile, boundaryFlux)

val solution = radialDiffusionStack forwardPropagate initialPSD

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


spline(lShellVec.zip(solution.head.toArray.toSeq))
hold()

(1 to tMax).foreach(l => {
  spline(lShellVec.zip(solution(l).toArray.toSeq))
})

unhold()

legend(DenseVector.tabulate[Double](tMax+1)(i =>
  if(i < nL) timeLimits._1+(rds.deltaT*i)
  else timeLimits._2).toArray.map(s => "t = "+"%3f".format(s)))

title("Variation of Phase Space Density f(L,t)")
xAxis("L")
yAxis("f(L,t)")
