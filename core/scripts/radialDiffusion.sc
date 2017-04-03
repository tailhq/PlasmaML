import breeze.stats.distributions._
import breeze.linalg._
import io.github.mandar2812.PlasmaML.RadialDiffusionSolver
import io.github.mandar2812.dynaml.probability._
import com.quantifind.charts.Highcharts._


val (nL,nT) = (20, 40)
val lMax = 15
val rds = new RadialDiffusionSolver((1.5, 6.5), (0.0, 10.0), nL, nT)

val ru = RandomVariable(new Uniform(0.0, 1.0))
val gau = GaussianRV(0.0, 1.5)
val rg = RandomVariable(new Beta(1.5, 2.5))

val randMatrix = (rv: RandomVariable[Double]) => (i: Int, j: Int) => DenseMatrix.tabulate[Double](i, j)((l,m) => rv.draw)

val randMatrixB = (rv: RandomVariable[Double]) => (i: Int, j: Int) => DenseMatrix.tabulate[Double](i, j)((l,m) => if (l == i-1) -rv.draw else if(l == 0) rv.draw else 0.0)

val (diffusionProfile, lossProfile, boundaryFlux) = (
  randMatrix(rg)(nL+1,nT),
  randMatrix(rg)(nL+1,nT),
  randMatrixB(rg)(nL+1,nT))

val radialDiffusionStack = rds.getComputationStack(diffusionProfile, lossProfile, boundaryFlux)

val solution = radialDiffusionStack forwardPropagate DenseVector.tabulate(nL+1)(l => 1.0)

spline(solution.map(_(0)))
hold()

(1 to lMax).foreach(l => {
  spline(solution.map(_(l)))
})

unhold()
