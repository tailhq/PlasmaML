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
mlpKernel.block_all_hyper_parameters

val whiteNoiseKernel = new DiracKernel(1.0)

OmniOSA.gridSize = 2

//Get test results for Linear GP-AR model
val resPolyAR = OmniOSA.buildAndTestGP(polynomialKernel, whiteNoiseKernel)

//Set solar wind speed and IMF Bz as exogenous variables
OmniOSA.setExogenousVars(List(24, 16), List(2,2))

//Reset kernel and noise to initial states
polynomialKernel.setoffset(0.5)
whiteNoiseKernel.setNoiseLevel(0.5)

//Get test results for a Linear GP-ARX model
val resPolyARX = OmniOSA.buildAndTestGP(polynomialKernel+tKernel+mlpKernel, whiteNoiseKernel)

//Compare with base line of the Persistence model
val resPer = DstPersistenceMOExperiment(0)

//Print the results out on the console
resPer.print()

resPolyAR.print()

resPolyARX.print()

