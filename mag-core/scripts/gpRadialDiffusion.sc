import breeze.linalg._
import breeze.stats.distributions.{ContinuousDistr, Gamma, Gaussian}
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder, MetaPipe}
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.models.gp.GPOperatorModel
import io.github.mandar2812.dynaml.probability.mcmc.HyperParameterMCMC
import io.github.mandar2812.dynaml.utils.ConfigEncoding

val (nL,nT) = (200, 50)

val lShellLimits = (1.0, 10.0)
val timeLimits = (0.0, 5.0)

val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

val (lShellVec, timeVec) = rds.stencil

val Kp = DataPipe((t: Double) =>
  if(t<= 0d) 2.5
  else if(t < 1.5) 2.5 + 4*t
  else if (t >= 1.5 && t< 3d) 8.5
  else if(t >= 3d && t<= 5d) 17.5 - 3*t
  else 2.5)

val rowSelectorRV = MultinomialRV(DenseVector.fill[Double](lShellVec.length)(1d/lShellVec.length.toDouble))

val baseNoiseLevel = 1.2
val mult = 0.8

/*
* Define parameters of radial diffusion system:
*
*  1) The diffusion field: dll
*  2) The particle injection process: q
*  3) The loss parameter: lambda
*
* Using the MagnetosphericProcessTrend class,
* we define each unknown process using a canonical
* parameterization of diffusion processes in the
* magnetosphere.
*
* For each process we must specify 4 parameters
* alpha, beta, a, b
* */

//Diffusion Field
val dll_alpha = 1d
val dll_beta = 10d
val dll_a = -9.325
val dll_b = 0.506

//Injection process
val q_alpha = 0d
val q_beta = 0d
val q_a = 0.0d
val q_b = 0.0d

//Loss Process
val lambda_alpha = math.pow(10d, -4)/2.4
val lambda_beta = 1d
val lambda_a = 2.5
val lambda_b = 0.18

val tau_alpha = 1/lambda_alpha
val tau_beta = -lambda_beta
val tau_a = -lambda_a
val tau_b = -lambda_b

//Create ground truth diffusion parameter functions
val dll = (l: Double, t: Double) => dll_alpha*math.pow(l, dll_beta)*math.pow(10, dll_a + dll_b*Kp(t))

val q = (l: Double, t: Double) => 0d

val lambda = (l: Double, t: Double) => lambda_alpha*math.pow(l, lambda_beta)*math.pow(10, lambda_a + lambda_b*Kp(t))

val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
val initialPSD = (l: Double) => math.sin(omega*(l - lShellLimits._1))*1E4

val initialPSDGT: DenseVector[Double] = DenseVector(lShellVec.map(l => initialPSD(l)).toArray)

//Create ground truth PSD data and corrupt it with statistical noise.
val groundTruth = rds.solve(q, dll, lambda)(initialPSD)
val ground_truth_matrix = DenseMatrix.horzcat(groundTruth.tail.map(_.asDenseMatrix.t):_*)
val measurement_noise = GaussianRV(0.0, 0.1)

val noise_mat = DenseMatrix.tabulate[Double](nL+1, nT)((_, _) => measurement_noise.draw)
val data: DenseMatrix[Double] = ground_truth_matrix + noise_mat

val gp_data: Seq[((Double, Double), Double)] = (0 until nT).map(col => {

  val rowS = rowSelectorRV.draw
  val (l, t) = (lShellVec(rowS), timeVec(col+1))
  ((l,t), data(rowS, col))
})


//Create the GP PDE model
val gpKernel = new SE1dDiffusionKernel(
  1.0, rds.deltaL, rds.deltaT, Kp)(
  (dll_alpha, dll_beta, dll_a, dll_b),
  (tau_alpha, -0.5, -0.5, 1.0)
)

val noiseKernel = new MAKernel(0.2)

noiseKernel.block_all_hyper_parameters
gpKernel.block(gpKernel.hyper_parameters.filter(_.contains("dll_")):_*)


implicit val dataT = DynaMLPipe.identityPipe[Seq[((Double, Double), Double)]]

val model = GPOperatorModel[Seq[((Double, Double), Double)], Double, SE1dDiffusionKernel](
  gpKernel, noiseKernel:*noiseKernel, DataPipe((_: (Double, Double)) => 0))(
  gp_data, gp_data.length)


//Create the MCMC sampler
val hyp = gpKernel.effective_hyper_parameters ++ noiseKernel.effective_hyper_parameters

val num_hyp = hyp.length

val proposal = MultGaussianRV(
  num_hyp)(
  DenseVector.zeros[Double](num_hyp),
  DenseMatrix.eye[Double](num_hyp))

val proposal_distr2 = MultStudentsTRV(num_hyp)(
  2.5, DenseVector.zeros[Double](num_hyp),
  DenseMatrix.eye[Double](num_hyp)*0.001)

val hyper_prior = {
  hyp.filter(_.contains("base::")).map(h => (h, new Gamma(1d, 1.5d))).toMap ++
    hyp.filterNot(_.contains("base::")).map(h => (h, new Gaussian(0d, 1.5d))).toMap
}

val mcmc = HyperParameterMCMC[model.type, ContinuousDistr[Double]](
  model, hyper_prior,
  proposal, 5000)


//Draw samples from the posteior
val samples = mcmc.iid(2000).draw

scatter(samples.map(c => (c("tau_alpha"), c("tau_beta"))))
hold()
scatter(Seq((tau_alpha, tau_beta)))
legend(Seq("Posterior Samples", "Ground Truth"))
title("Posterior Samples:- Tau(l, t) = alpha*l^(beta)*10^(a*Kp(t) + b)")
xAxis("Tau: alpha")
yAxis("Tau: beta")
unhold()


scatter(samples.map(c => (c("tau_a"), c("tau_b"))))
hold()
scatter(Seq((tau_a, tau_b)))
legend(Seq("Posterior Samples", "Ground Truth"))
title("Posterior Samples:- Tau(l, t) = alpha*l^(beta)*10^(a*Kp(t) + b)")
xAxis("Tau: a")
yAxis("Tau: b")
unhold()
