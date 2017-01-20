//DynaML imports
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
//Import Omni programs
import io.github.mandar2812.PlasmaML.omni._

val polynomialKernel = new PolynomialKernel(1, 0.5)
polynomialKernel.block("degree")

implicit val ev = VectorField(OmniOSA.modelOrders._1+OmniOSA.modelOrders._2.sum)

val tKernel = new TStudentKernel(1.0)

val mlpKernel = new MLPKernel(0.909, 0.909)
//mlpKernel.block_all_hyper_parameters

val whiteNoiseKernel = new DiracKernel(0.5)

OmniOSA.gridSize = 2

//Set model validation data set ranges
OmniOSA.validationDataSections ++= Stream(
  ("2013/03/17/07", "2013/03/18/10"),
  ("2011/10/24/20", "2011/10/25/14"))

OmniOSA.clearExogenousVars()
//Get test results for Linear GP-AR model
//with a mean function given by the persistence
//model
val resPolyAR = OmniOSA.buildAndTestGP(
  mlpKernel+tKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)

//Set solar wind speed and IMF Bz as exogenous variables
OmniOSA.setExogenousVars(List(24, 16), List(2,2))

//Reset kernel and noise to initial states
polynomialKernel.setoffset(0.5)
whiteNoiseKernel.setNoiseLevel(0.5)
OmniOSA.gridSize = 2
//Get test results for a GP-ARX model
//with a mean function given by the persistence
//model
val resPolyARX = OmniOSA.buildAndTestGP(
  tKernel+mlpKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)

OmniOSA.clearExogenousVars()
//Compare with base line of the Persistence model
val resPer = DstPersistenceMOExperiment(0)

//Print the results out on the console
resPer.print()

resPolyAR.print()

resPolyARX.print()

