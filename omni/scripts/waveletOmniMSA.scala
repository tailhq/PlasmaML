import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes.Encoder

OmniMSA.quietTimeSegment = ("2014/01/01/20", "2014/01/06/20")
OmniMultiOutputModels.exogenousInputs = List(24,16)
val numVars = OmniMultiOutputModels.exogenousInputs.length + 1
OmniMultiOutputModels.orderFeat = 3
OmniMultiOutputModels.orderTarget = 2

val num_features = if(OmniMultiOutputModels.deltaT.isEmpty) {
  (1 until numVars).map(_ => math.pow(2.0, OmniMultiOutputModels.orderFeat)).sum.toInt +
    math.pow(2d, OmniMultiOutputModels.orderFeat).toInt
} else {
  OmniMultiOutputModels.deltaT.sum
}

DstMSAExperiment.gridSize = 2
DstMSAExperiment.gridStep = 0.75
DstMSAExperiment.useLogScale = true
DstMSAExperiment.globalOpt = "CSA"
DstMSAExperiment.it = 15

implicit val ev = VectorField(num_features)

val gsm_encoder = Encoder(
  (config: Map[String, Double]) => (DenseVector.fill[Double](num_features)(config("c")), DenseVector.fill[Double](num_features)(config("s"))),
  (centerAndScale: (DenseVector[Double], DenseVector[Double])) => Map("c" -> centerAndScale._1(0), "s"-> centerAndScale._2(0))
)

val (center, scale) = gsm_encoder(Map("c" -> 1.5, "s" -> 10.5))

val gsm = GaussianSpectralKernel[DenseVector[Double]](center, scale, gsm_encoder)

val cubicSplineKernel = new CubicSplineKernel[DenseVector[Double]](1.5)

val d = new DiracKernel(0.05)
d.block_all_hyper_parameters

val tKernel = new TStudentKernel(0.01/*0.5+1.0/num_features*/)
tKernel.block_all_hyper_parameters

val mlpKernel = new MLPKernel(1.2901485870065708, 73.92009461973996)


//val (model, scales) = OmniMSA.train(mlpKernel+tKernel, d, 4, 0.5, false, DynaMLPipe.identityPipe[Features])

val metricsMT = DstMSAExperiment(mlpKernel+cubicSplineKernel+tKernel, d, 3, 2, useWavelets = false)
metricsMT.print

OmniMultiOutputModels.exogenousInputs = List(24, 16)

OmniMultiOutputModels.neuronCounts = List(12, 8, 6)
OmniMultiOutputModels.activations = List(MagicSELU, MagicSELU, MagicSELU, VectorLinear)

DstMSAExperiment.learningRate = 0.001
DstMSAExperiment.momentum = 0.005
DstMSAExperiment.it = 7000
DstMSAExperiment.reg = 0.0000001

val metricsW = DstMSAExperiment(4, 2)
