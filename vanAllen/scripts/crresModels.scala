import io.github.mandar2812.PlasmaML.vanAllen.CRRESKernel

// Test a GP model
val waveletF = (x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0)

CRRESTest(
  new WaveletKernel(waveletF)(1.1) * new RBFKernel(1.5),
  new DiracKernel(0.6), 1000, 2000,
  1, 0.02)

CRRESTest(
  new CRRESKernel(1.02, 1.5),
  new DiracKernel(0.94), 2000, 1000,
  1, 0.02)


// Test a feed forward neural network
CRRESTest(
  1, List("tansig", "linear"), List(16),
  7000, 3000, 0.1, 80,
  0.4, 0.00, 0.75
)
