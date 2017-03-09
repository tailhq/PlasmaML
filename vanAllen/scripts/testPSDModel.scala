import breeze.linalg.DenseVector
import com.jmatio.types.MLDouble
import io.github.mandar2812.dynaml.DynaMLPipe.identityPipe
import io.github.mandar2812.dynaml.analysis.{DifferentiableMap, PartitionedVectorField, PushforwardMap, VectorField}
import io.github.mandar2812.dynaml.dataformat.MAT
import io.github.mandar2812.dynaml.kernels.{DiracKernel, MLPKernel, RBFKernel}
import io.github.mandar2812.dynaml.models.gp.{GPRegression, WarpedGP}
import io.github.mandar2812.dynaml.optimization.GridSearch
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder}

import scala.util.Random

val psdProfile = """/Users/mandar/Google Drive/CWI/PSD_data/psdprofiles_rbsp_mu_700.0_k_0.1.mat"""

val psdData = (MAT.read > MAT.content)(psdProfile)

val psd = psdData("PSD_arr").asInstanceOf[MLDouble]
val time = psdData("Time_arr").asInstanceOf[MLDouble]
val l_star = psdData("Lstar_arr").asInstanceOf[MLDouble]

val Array(rows, cols) = psd.getDimensions

val data = for(rowIndex <- 0 until rows; colIndex <- 0 until cols)
  yield (
    DenseVector(time.getReal(rowIndex, 0).toDouble, l_star.getReal(0, colIndex).toDouble),
    psd.getReal(rowIndex, colIndex).toDouble)

val filteredData =
  Random.shuffle(
    data
    .filterNot(_._2.isNaN)
    .map(c => (c._1 - DenseVector(735143.0,0.0), c._2))
    .toStream)

val (training, test) = (filteredData.take(1000), filteredData.takeRight(1000))

implicit val ev = VectorField(2)
val kernel = new MLPKernel(1.25, 1.5)
val noise = new DiracKernel(1.0)
val gpModel = new GPRegression(kernel, noise, training)



implicit val detImpl = identityPipe[Double]

val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => math.exp(-x)),
  DifferentiableMap(
    (x: Double) => -math.log(x),
    (x: Double) => -1.0/x)
)

val h1: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => -math.log(x)),
  DifferentiableMap(
    (x: Double) => math.exp(-x),
    (x: Double) => -math.exp(-x))
)


implicit val pVF = PartitionedVectorField(1000, 1000)
implicit val t = Encoder(
  identityPipe[Seq[(DenseVector[Double], Double)]],
  identityPipe[Seq[(DenseVector[Double], Double)]])



val startConf = kernel.effective_state ++ noise.effective_state


val wGP = new WarpedGP(gpModel)(h)


val gs = new GridSearch[wGP.type](wGP).setGridSize(3).setStepSize(0.25).setLogScale(true)

val (optModel, conf) = gs.optimize(startConf, Map())


val res = optModel.test(test).map(c => (c._3, c._2))
