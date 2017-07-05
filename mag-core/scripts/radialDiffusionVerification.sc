import breeze.linalg._
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts.AxisType
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RadialDiffusion


val (nL,nT) = (10, 10)


val bins = List(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val theta = 0.006
val alpha = 0.01 + theta*math.pow(omega*lShellLimits._2, 2.0)

val fl = (l: Double, _: Double) => math.sin(omega*(l - lShellLimits._1))
val ft = (_: Double, t: Double) => math.exp(-alpha*t) + 1.0

val referenceSolution = (l: Double, t: Double) => fl(l,t)*ft(l,t)

val radialDiffusionSolver = (binsL: Int, binsT: Int) => new RadialDiffusion(lShellLimits, timeLimits, binsL, binsT)

val q = (l: Double, t: Double) => alpha*fl(l,t)
val dll = (l: Double, _: Double) => theta*l*l
val loss = (l: Double, _: Double) => alpha - math.pow(l*omega, 2.0)*theta

val initialPSD = (l: Double) => referenceSolution(l, 0.0)


//Perform verification of errors for constant nL

val lossesTime = bins.map(bT => {

  val rds = radialDiffusionSolver(nL, bT)

  println("Solving for delta T = "+rds.deltaT)

  val lShellVec = DenseVector.tabulate[Double](nL+1)(i =>
    if(i < nL) lShellLimits._1+(rds.deltaL*i)
    else lShellLimits._2).toArray.toSeq

  val timeVec = DenseVector.tabulate[Double](bT+1)(i =>
    if(i < bT) timeLimits._1+(rds.deltaT*i)
    else timeLimits._2).toArray.toSeq

  println("\tGenerating neural computation stack & computing solution")

  val solution = rds.solve(q, dll, loss)(initialPSD)

  val referenceSol = timeVec.map(t =>
    DenseVector(lShellVec.map(lS => referenceSolution(lS, t)).toArray))

  println("\tCalculating RMSE with respect to reference solution\n")

  (rds.deltaT, RadialDiffusion.error(referenceSol)(solution))

})


val lossesSpace = bins.map(bL => {

  val rds = radialDiffusionSolver(bL, nT)

  println("Solving for delta L = "+rds.deltaL)
  val lShellVec = DenseVector.tabulate[Double](bL+1)(i =>
    if(i < bL) lShellLimits._1+(rds.deltaL*i)
    else lShellLimits._2).toArray.toSeq

  val timeVec = DenseVector.tabulate[Double](nT+1)(i =>
    if(i < nT) timeLimits._1+(rds.deltaT*i)
    else timeLimits._2).toArray.toSeq

  println("\tGenerating neural computation stack & computing solution")

  val solution = rds.solve(q, dll, loss)(initialPSD)

  val referenceSol = timeVec.map(t =>
    DenseVector(lShellVec.map(lS => referenceSolution(lS, t)).toArray))

  println("\tCalculating RMSE with respect to reference solution\n")

  (rds.deltaL, RadialDiffusion.error(referenceSol)(solution))

})



spline(lossesTime)
title("Forward Solver Error")
xAxisType(AxisType.logarithmic)
xAxis("delta T")
yAxis("RMSE")


spline(lossesSpace)
title("Forward Solver Error")
xAxisType(AxisType.logarithmic)
xAxis("delta L")
yAxis("RMSE")
