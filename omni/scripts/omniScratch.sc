//DynaML imports
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler
//Import Omni programs
import io.github.mandar2812.PlasmaML.omni._

OmniOSA.setTarget(40, 10)
//OmniOSA.setExogenousVars(List(24, 5), List(24, 5))
OmniOSA.clearExogenousVars()
val trainAutoEnc = DataPipe(
  (d: Stream[(OmniOSA.Features, Double)]) => d.map(_._1)) > DataPipe((d: Stream[OmniOSA.Features]) => {

  val layerSizes = List(OmniOSA.input_dimensions, 8, 4, 8, OmniOSA.input_dimensions)
  val activations = (1 until layerSizes.length).map(_ => VectorTansig).toList
  val autoenc = GenericAutoEncoder(layerSizes, List(VectorTansig, VectorSigmoid, VectorSigmoid, VectorTansig))

  autoenc
    .optimizer
    .setNumIterations(5000)
    .setStepSize(0.02)
    .momentum_(0.75)
    .setRegParam(0.0001)
    .setMiniBatchFraction(1.0)

  autoenc.learn(d)
  autoenc
})

val pipe = OmniOSA.dataPipeline > DataPipe(trainAutoEnc, identityPipe[(GaussianScaler, GaussianScaler)])

val (autoenc, scalers) =
  pipe.run(
    OmniOSA.trainingDataSections ++
      Stream(
        ("2012/01/10/00", "2012/12/28/23"),
        ("2013/01/10/00", "2012/12/28/23"),
        ("2014/01/10/00", "2014/12/28/23")))
