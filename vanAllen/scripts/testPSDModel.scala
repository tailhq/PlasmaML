import breeze.linalg.DenseVector
import com.jmatio.types.MLDouble
import io.github.mandar2812.dynaml.DynaMLPipe.identityPipe
import io.github.mandar2812.dynaml.analysis.{DifferentiableMap, PartitionedVectorField, PushforwardMap, VectorField}
import io.github.mandar2812.dynaml.dataformat.MAT
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.{GPRegression, WarpedGPModel}
import io.github.mandar2812.dynaml.models.sgp.AbstractSkewGPModel
import io.github.mandar2812.dynaml.optimization.{CoupledSimulatedAnnealing, GridSearch}
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
    math.log(psd.getReal(rowIndex, colIndex).toDouble))

val filteredData =
  Random.shuffle(
    data
    .filterNot(_._2.isNaN)
    .map(c => (c._1 - DenseVector(735143.0,0.0), c._2))
    .toStream)

val (training, test) = (filteredData.take(1000), filteredData.takeRight(1000))

implicit val trans = DataPipe((s: Stream[(DenseVector[Double], Double)]) => s.toSeq)
implicit val ev = VectorField(2)

val tKernel = new TStudentKernel(0.01)
//tKernel.block_all_hyper_parameters
val mlpKernel = new MLPKernel(1.25, 1.5)

val enc = Encoder[Map[String, Double], (DenseVector[Double], DenseVector[Double])](
  (c: Map[String, Double]) => {
    val (centerConf, scaleConf) = (
      c.filter(k => k._1.contains("c")).map(k => (k._1.split("_").last.toInt, k._2)),
      c.filter(k => k._1.contains("s")).map(k => (k._1.split("_").last.toInt, k._2)))

    (
      DenseVector.tabulate[Double](centerConf.size)(i => centerConf(i)),
      DenseVector.tabulate[Double](scaleConf.size)(i => scaleConf(i)))
  },
  (vecs: (DenseVector[Double], DenseVector[Double])) => {
    vecs._1.toArray.zipWithIndex.map((c) => ("c_"+c._2, c._1)).toMap ++
      vecs._2.toArray.zipWithIndex.map((c) => ("s_"+c._2, c._1)).toMap
  }
)


val gaussianSMKernel = GaussianSMKernel(DenseVector(2.5, 2.5), DenseVector(0.5, 10.0), enc)


val kernel = mlpKernel + tKernel
val noise = new DiracKernel(1.0)


val gpModel = new GPRegression(gaussianSMKernel, noise, training)

val sgpModel = AbstractSkewGPModel(kernel, noise, DataPipe((x: DenseVector[Double]) => 0.0), 1.5, 0.5)(training)

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



val startConf = kernel.effective_state ++ noise.effective_state ++ Map("skewness" -> 1.5, "cutoff" -> 0.5)


val wGP = new WarpedGPModel(gpModel)(h)


val gs = new GridSearch[sgpModel.type ](sgpModel).setGridSize(2).setStepSize(0.45).setLogScale(true)

val csa =
  new CoupledSimulatedAnnealing[sgpModel.type](sgpModel).setGridSize(1).setStepSize(0.5).setLogScale(true).setMaxIterations(40)

val (optModel, conf) = csa.optimize(startConf, Map())


val res = optModel.test(test).map(c => (c._3, c._2))
