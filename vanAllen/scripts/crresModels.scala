import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.vanAllen.CRRESKernel
import io.github.mandar2812.dynaml.kernels.{CompositeCovariance, DiracKernel, LocalSVMKernel}
import io.github.mandar2812.dynaml.pipes.DataPipe

// Test a GP model
val waveletF = (x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0)

CRRESTest(
  new WaveletKernel(waveletF)(1.1) * new RBFKernel(1.5),
  new DiracKernel(0.6), 1000, 2000,
  1, 0.02)

val crresKern: CRRESKernel = new CRRESKernel(1.0, 1.48, 1.0, 1.0)

crresKern.blocked_hyper_parameters =
  crresKern.rbfK.hyper_parameters ++ crresKern.waveletK.hyper_parameters

val noiseKern:DiracKernel = new DiracKernel(0.74)

noiseKern.blocked_hyper_parameters = noiseKern.hyper_parameters

CRRESTest(
  crresKern,
  noiseKern, 2000, 1000,
  2, 0.2)


val waveletKern: LocalSVMKernel[DenseVector[Double]] = new WaveletKernel(waveletF)(1.1)

val rbfKern: LocalSVMKernel[DenseVector[Double]] = new RBFKernel(1.5)



val crresKern1: CompositeCovariance[DenseVector[Double]] = (waveletKern * rbfKern)

crresKern1.blocked_hyper_parameters = waveletKern.hyper_parameters ++ rbfKern.hyper_parameters

CRRESTest(
  crresKern1,
  new DiracKernel(0.84), 2000, 2000,
  2, 0.02)


// Test a feed forward neural network
CRRESTest(
  1, List("logsig", "linear"), List(24),
  7000, 3000, 0.1, 100,
  0.4, 0.00, 0.75
)
