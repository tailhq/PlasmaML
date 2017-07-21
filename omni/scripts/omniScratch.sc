//DynaML imports
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import io.github.mandar2812.dynaml.probability.distributions.UnivariateGaussian
import io.github.mandar2812.dynaml.probability.mcmc._
import io.github.mandar2812.dynaml.utils.GaussianScaler
//Import Omni programs
import io.github.mandar2812.PlasmaML.omni._
import com.quantifind.charts.Highcharts._

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


val polynomialKernel = new PolynomialKernel(1, 0.5)
polynomialKernel.block("degree")

implicit val ev = VectorField(OmniOSA.input_dimensions)

val tKernel = new TStudentKernel(0.01)

val rbfKernel = new RBFKernel(1.7)

val mlpKernel = new MLPKernel(1.5, 0.75)

val whiteNoiseKernel = new DiracKernel(0.2)
whiteNoiseKernel.block_all_hyper_parameters

//Use best performing model for generating predictions
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0
OmniOSA.globalOpt = "GS"
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 16), List(1,3))
mlpKernel.setw(1.2901485870065708)
mlpKernel.setoffset(73.92009461973996)
OmniOSA.gridSize = 1
OmniOSA.gridStep = 0.0

val predictPipeline = OmniOSA.dataPipeline > OmniOSA.gpTrain(
    tKernel+mlpKernel,
    whiteNoiseKernel,
    OmniOSA.meanFuncPersistence)

val predictPipelineSGP = OmniOSA.dataPipeline > OmniOSA.sgpTrain(
  tKernel+mlpKernel, whiteNoiseKernel,
  -4.4, -0.01, OmniOSA.meanFuncPersistence)


val (model, scaler) = predictPipeline(OmniOSA.trainingDataSections)

val (sgp_model, scaler_sgp) = predictPipelineSGP(OmniOSA.trainingDataSections)

val num_hyp = model._hyper_parameters.length
val num_hyp_sgp = sgp_model._hyper_parameters.length

val mcmc = new AdaptiveHyperParameterMCMC[model.type, ContinuousDistr[Double]](
  model, model._hyper_parameters.map(h => (h, new LogNormal(0.0, 2.0))).toMap,
  500)



val samples = mcmc.iid(2000).draw

val sgp_hyper_prior: Map[String, ContinuousDistr[Double]] = {
  sgp_model._hyper_parameters.filterNot(h => h.contains("skewness") || h.contains("cutoff")).map(h => (h, new LogNormal(0.0, 2.0))).toMap ++
  Map("skewness" -> Gaussian(0d, 2d), "cutoff" -> Gaussian(0d, 2d))
}

val mcmc_sgp = new AdaptiveHyperParameterMCMC[sgp_model.type, ContinuousDistr[Double]](
  sgp_model, sgp_hyper_prior,
  500)


val samples_sgp = mcmc_sgp.iid(2000).draw


scatter(samples.map(c => (c("MLPKernel@a668c21/w"), c("MLPKernel@a668c21/b"))))
title("Posterior Samples")
xAxis("MLP Kernel: w")
yAxis("MLP Kernel: b")


scatter(samples_sgp.map(c => (c("MLPKernel@a668c21/w"), c("MLPKernel@a668c21/b"))))
title("Posterior Samples")
xAxis("MLP Kernel: w")
yAxis("MLP Kernel: b")

