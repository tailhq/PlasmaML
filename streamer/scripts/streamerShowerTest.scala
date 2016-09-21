import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.streamer.StreamerShowerEmulator
import io.github.mandar2812.dynaml.kernels._

import io.github.mandar2812.dynaml.analysis.VectorField

val num_features = 5
implicit val ev = VectorField(num_features)


val stK = new TStudentKernel(2.2)
val sqExpK = new SEKernel(1.5, 0.7)
sqExpK.blocked_hyper_parameters = List("amplitude")
val linearK = new PolynomialKernel(1, 0.0)
linearK.blocked_hyper_parameters = linearK.hyper_parameters

val cauK = new CauchyKernel(2.4)

val n = new DiracKernel(0.1)
n.blocked_hyper_parameters = n.hyper_parameters

StreamerShowerEmulator.globalOpt = "GS"
StreamerShowerEmulator.trainingSize = 3000
val resStreamerGP = StreamerShowerEmulator(cauK + linearK + sqExpK, n, 2, 0.2, false, 15)

resStreamerGP.print


val sqExpK = new SEKernel(1.5, 0.7)
sqExpK.blocked_hyper_parameters = List("amplitude")

val resStreamerSTP = StreamerShowerEmulator(linearK + sqExpK, n, 10.0, 2, 0.2, false, 15)

resStreamerSTP.print


