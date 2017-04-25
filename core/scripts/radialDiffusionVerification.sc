import breeze.linalg._
import com.quantifind.charts.Highcharts._
import com.quantifind.charts.highcharts.AxisType
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RadialDiffusion


val (nL,nT) = (10, 10)


val bins = List(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

val lShellLimits = (1.0, 5.0)
val timeLimits = (0.0, 5.0)

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val theta = 0.06
val alpha = 0.005 + theta*math.pow(omega*lShellLimits._2, 2.0)

val referenceSolution = (l: Double, t: Double) => math.sin(omega*(l - lShellLimits._1))*(math.exp(-alpha*t) + 1.0)

val radialDiffusionSolver = (binsL: Int, binsT: Int) => new RadialDiffusion(lShellLimits, timeLimits, binsL, binsT)

val boundFlux = (l: Double, t: Double) => {
  if(l == lShellLimits._1 || l == lShellLimits._2) referenceSolution(l, t) else 0.0
}

val dll = (l: Double, _: Double) => theta*l*l
val q = (l: Double, _: Double) => alpha - math.pow(l*omega, 2.0)*theta

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

  val solution = rds.solve(q, dll, boundFlux)(initialPSD)

  val referenceSol = timeVec.map(t =>
    DenseVector(lShellVec.map(lS => referenceSolution(lS, t)).toArray))

  println("\tCalculating RMSE with respect to reference solution\n")
  val error = math.sqrt(
    solution.zip(referenceSol).map(c => math.pow(norm(c._1 - c._2, 2.0), 2.0)).sum/(nL+1.0)*(bT+1.0)
  )

  (rds.deltaT, error)

})

spline(lossesTime)
title("Forward Solver Error")
xAxisType(AxisType.logarithmic)
xAxis("delta T")
yAxis("RMSE")


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

  val solution = rds.solve(q, dll, boundFlux)(initialPSD)

  val referenceSol = timeVec.map(t =>
    DenseVector(lShellVec.map(lS => referenceSolution(lS, t)).toArray))

  println("\tCalculating RMSE with respect to reference solution\n")
  val error = math.sqrt(
    solution.zip(referenceSol).map(c => math.pow(norm(c._1 - c._2, 2.0), 2.0)).sum/((bL+1.0)*(nT+1d))
  )

  (rds.deltaL, error)

})


spline(lossesSpace)
title("Forward Solver Error")
xAxisType(AxisType.logarithmic)
xAxis("delta L")
yAxis("RMSE")
