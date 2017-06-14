import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels.{DiracKernel, MLPKernel, TStudentKernel}
import io.github.mandar2812.dynaml.models.neuralnets._

OmniMSA.quietTimeSegment = ("2014/01/01/20", "2014/01/02/20")
OmniMultiOutputModels.exogenousInputs = List(24,16)
val numVars = OmniMultiOutputModels.exogenousInputs.length + 1
OmniMultiOutputModels.orderFeat = 3
OmniMultiOutputModels.orderTarget = 2

val num_features = if(OmniMultiOutputModels.deltaT.isEmpty) {
  (1 until numVars).map(_ => math.pow(2.0, OmniMultiOutputModels.orderFeat)).sum.toInt +
    math.pow(2d, OmniMultiOutputModels.orderTarget).toInt
} else {
  OmniMultiOutputModels.deltaT.sum
}

DstMSAExperiment.gridSize = 4

implicit val ev = VectorField(num_features)

val d = new DiracKernel(0.05)
d.block_all_hyper_parameters

val tKernel = new TStudentKernel(2.51/*0.5+1.0/num_features*/)
//tKernel.block_all_hyper_parameters

val mlpKernel = new MLPKernel(15.0, 5.5)


//val (model, scales) = OmniMSA.train(mlpKernel+tKernel, d, 4, 0.5, false, DynaMLPipe.identityPipe[Features])

val metricsMT = DstMSAExperiment(mlpKernel+tKernel, d, 3, 2, useWavelets = true)
metricsMT.print

OmniMultiOutputModels.exogenousInputs = List(24, 16)

OmniMultiOutputModels.neuronCounts = List(6, 4)
OmniMultiOutputModels.activations = List(MagicSELU, MagicSELU, VectorLinear)

DstMSAExperiment.learningRate = 0.075
DstMSAExperiment.momentum = 0.25
DstMSAExperiment.it = 10000
DstMSAExperiment.reg = 0.00001

val metricsW = DstMSAExperiment(3, 2)
