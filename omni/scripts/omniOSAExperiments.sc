//DynaML imports
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
//Import Omni programs
import io.github.mandar2812.PlasmaML.omni._

val polynomialKernel = new PolynomialKernel(1, 0.5)
polynomialKernel.block("degree")

implicit val ev = VectorField(OmniOSA.input_dimensions)

val tKernel = new TStudentKernel(0.01)
tKernel.block_all_hyper_parameters

val rbfKernel = new RBFKernel(1.7)

val mlpKernel = new MLPKernel(1.0, 1.0)

val whiteNoiseKernel = new DiracKernel(0.2)
whiteNoiseKernel.block_all_hyper_parameters

OmniOSA.gridSize = 2
OmniOSA.gridStep = 0.2
OmniOSA.globalOpt = "ML-II"
OmniOSA.maxIterations = 250

//Set model validation data set ranges
/*OmniOSA.validationDataSections ++= Stream(
  ("2013/03/17/07", "2013/03/18/10"),
  ("2011/10/24/20", "2011/10/25/14"))*/

OmniOSA.clearExogenousVars()
//Get test results for Linear GP-AR model
//with a mean function given by the persistence
//model
OmniOSA.setTarget(40, 10)
val resPolyAR = OmniOSA.buildAndTestGP(
  mlpKernel+tKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)


OmniOSA.clearExogenousVars()
//Set solar wind speed and IMF Bz as exogenous variables
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 16), List(2,2))
//Reset kernel and noise to initial states
mlpKernel.setw(1.0)
mlpKernel.setoffset(1.0)
OmniOSA.gridSize = 2
//Get test results for a GP-ARX model
//with a mean function given by the persistence
//model
val resPolyARX = OmniOSA.buildAndTestGP(
  tKernel+mlpKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)


//Compare with base line of the Persistence model
val resPer = DstPersistenceMOExperiment(0)

OmniOSA.clearExogenousVars()

mlpKernel.setw(1.0)
mlpKernel.setoffset(1.0)
OmniOSA.gridSize = 3
OmniOSA.globalOpt = "ML-II"
OmniOSA.modelType_("GP-NARMAX")
OmniOSA.setTarget(40, 6)
OmniOSA.setExogenousVars(List(24, 15, 16, 28), List(4), changeModelType = false)

val resPolyNM = OmniOSA.buildAndTestGP(
  tKernel+mlpKernel,
  whiteNoiseKernel,
  OmniOSA.meanFuncPersistence)

//Print the results out on the console
resPer.print()
resPolyAR.print()
resPolyARX.print()
resPolyNM.print()

OmniOSA.clearExogenousVars()
