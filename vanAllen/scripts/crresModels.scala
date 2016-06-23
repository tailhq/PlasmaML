
// Test a GP model
val waveletF = (x: Double) => math.cos(1.75*x)*math.exp(-1*x*x/2.0)

CRRESTest(
  new WaveletKernel(waveletF)(1.1) * new RBFKernel(1.5),
  new DiracKernel(0.6), 1000, 2000,
  1, 0.02)

// Test a feed forward neural network
CRRESTest(
  1, List("tansig", "linear"), List(12),
  7000, 3000, 0.2, 120,
  0.5, 0.00, 0.75
)
