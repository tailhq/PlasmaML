package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector, diag, norm}
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.SGRadialDiffusionModel.GaussianQuadrature
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
  * Inverse inference over plasma radial diffusion parameters.
  *
  * @param Kp A function which returns the Kp value for a given
  *           time coordinate. Must be cast as a [[DataPipe]]
  *
  * @param dll_params A [[Tuple4]] containing the diffusion field
  *                   parameters. See [[io.github.mandar2812.PlasmaML.utils.MagConfigEncoding]] and
  *                   [[MagnetosphericProcessTrend]].
  *
  * @param tau_params A [[Tuple4]] containing the loss process parameters.
  *
  * @param covariance A kernel function representing the covariance of
  *                   the Phase Space Density at a pair of space time locations.
  *
  * @param noise_psd A kernel function representing the measurement noise of the
  *                  Phase Space Density at a pair of space time locations.
  *
  * @param psd_data A Stream of space time locations and measured PSD values.
  *
  * @param ghost_points A collection of "ghost" points on which Particle diffusion is computed
  *                     and its dependence on PSD is enforced with square loss.
  *
  * @param basis A basis function expansion for the PSD, as an instance
  *              of [[PSDBasis]].
  * */
class SGRadialDiffusionModel(
  val Kp: DataPipe[Double, Double],
  dll_params: (Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double),
  q_params: (Double, Double, Double, Double))(
  val covariance: LocalScalarKernel[(Double, Double)],
  val noise_psd: DiracTuple2Kernel,
  val psd_data: Stream[((Double, Double), Double)],
  val basis: PSDBasis,
  val lShellDomain: (Double, Double),
  val timeDomain: (Double, Double),
  val quadrature: GaussianQuadrature = SGRadialDiffusionModel.eightPointGaussLegendre)
  extends GloballyOptimizable {

  protected val logger: Logger = Logger.getLogger(this.getClass)

  val baseCovID: String = "base::"+covariance.toString.split("\\.").last

  val baseNoiseID: String = "base_noise::"+noise_psd.toString.split("\\.").last

  val diffusionField: MagTrend = new MagTrend(Kp, "dll")

  val lossTimeScale: MagTrend = new MagTrend(Kp, "tau")

  val injection_process: MagTrend = new MagTrend(Kp, "Q")

  val num_observations: Int = psd_data.length

  val (ghost_points, quadrature_weights): (Seq[(Double, Double)], DenseMatrix[Double]) = {

    val (l_nodes, l_weights) = quadrature.scale(lShellDomain._1, lShellDomain._2)

    val (t_nodes, t_weights) = quadrature.scale(timeDomain._1, timeDomain._2)

    val (points, weights) = utils.combine(Seq(l_nodes.zip(l_weights), t_nodes.zip(t_weights))).map(s => {

      val point = (s.head._1, s.last._1)
      val weight = s.head._2*s.last._2

      (point, weight)
    }).unzip


    (points, diag(DenseVector(weights.toArray)))
  }

  val num_colocation_points: Int = ghost_points.length

  val psd_mean: Double = psd_data.map(_._2).sum/num_observations

  val psd_std: Double  = math.sqrt(
    psd_data.map(p => p._2 - psd_mean).map(p => math.pow(p, 2d)).sum/(num_observations - 1)
  )

  private lazy val targets = DenseVector(psd_data.map(_._2).toArray)

  private val (covStEncoder, noiseStEncoder) = (
    BasisFuncRadialDiffusionModel.stateEncoder(baseCovID),
    BasisFuncRadialDiffusionModel.stateEncoder(baseNoiseID)
  )

  private val designMatrixFlow = BasisFuncRadialDiffusionModel.metaDesignMatFlow(basis)

  lazy val phi = designMatrixFlow(psd_data.map(_._1))

  private lazy val (aMat, b) = psd_data.map(p => {
    val ph = basis(p._1)
    val y = p._2
    (ph*ph.t, ph*y)
  }).reduceLeft((x, y) => (x._1+y._1, x._2+y._2))

  private val I = DenseMatrix.eye[Double](aMat.rows)

  def _operator_hyper_parameters: List[String] = operator_hyper_parameters

  protected val operator_hyper_parameters: List[String] = {

    val dll_hyp = diffusionField.transform.keys
    val tau_hyp = lossTimeScale.transform.keys
    val q_hyp = injection_process.transform.keys

    List(
      dll_hyp._1, dll_hyp._2, dll_hyp._3, dll_hyp._4,
      tau_hyp._1, tau_hyp._2, tau_hyp._3, tau_hyp._4,
      q_hyp._1, q_hyp._2, q_hyp._3, q_hyp._4
    )
  }

  /**
    * Stores the value of the operator parameters
    * as a [[Map]].
    * */
  protected var operator_state: Map[String, Double] = {
    val dll_hyp = diffusionField.transform.keys
    val tau_hyp = lossTimeScale.transform.keys
    val q_hyp = injection_process.transform.keys

    Map(
      dll_hyp._1 -> dll_params._1, dll_hyp._2 -> dll_params._2,
      dll_hyp._3 -> dll_params._3, dll_hyp._4 -> dll_params._4,
      tau_hyp._1 -> tau_params._1, tau_hyp._2 -> tau_params._2,
      tau_hyp._3 -> tau_params._3, tau_hyp._4 -> tau_params._4,
      q_hyp._1 -> q_params._1, q_hyp._2 -> q_params._2,
      q_hyp._3 -> q_params._3, q_hyp._4 -> q_params._4
    )
  }

  override var hyper_parameters: List[String] =
    covariance.hyper_parameters.map(h => baseCovID+"/"+h) ++
      noise_psd.hyper_parameters.map(h => baseNoiseID+"/"+h) ++
      operator_hyper_parameters


  /**
    * A Map which stores the current state of the system.
    * */
  override protected var current_state: Map[String, Double] =
    covStEncoder(covariance.state) ++
      noiseStEncoder(noise_psd.state) ++
      operator_state


  var blocked_hyper_parameters: List[String] =
    covariance.blocked_hyper_parameters.map(h => baseCovID+"/"+h) ++
      noise_psd.blocked_hyper_parameters.map(h => baseNoiseID+"/"+h)

  var reg: Double = 1d

  var (regObs, regCol): (Double, Double) = (1d, 1d)

  def block(hyp: String*): Unit = {

    val (blocked_cov_hyp, _) = hyp.partition(c => c.contains(baseCovID) || c.contains(baseNoiseID))

    val proc_cov_hyp = blocked_cov_hyp.filter(_.contains(baseCovID))
      .map(h => h.replace(baseCovID, "").tail)

    val proc_noise_hyp = blocked_cov_hyp.filter(_.contains(baseNoiseID))
      .map(h => h.replace(baseNoiseID, "").tail)


    covariance.block(proc_cov_hyp:_*)
    noise_psd.block(proc_noise_hyp:_*)
    blocked_hyper_parameters = hyp.toList

  }

  def block_++(h: String*): Unit = block(blocked_hyper_parameters.union(h):_*)

  def effective_hyper_parameters: List[String] =
    hyper_parameters.filterNot(h => blocked_hyper_parameters.contains(h))

  def effective_state: Map[String, Double] = _current_state.filterKeys(effective_hyper_parameters.contains)

  def setState(h: Map[String, Double]): Unit = {

    require(
      effective_hyper_parameters.forall(h.contains),
      "All Hyper-parameters must be contained in state assignment")

    val base_kernel_state = h.filterKeys(
      _.contains(baseCovID)).map(
      c => (c._1.replace(baseCovID, "").tail, c._2)
    )

    val base_noise_state = h.filterKeys(
      _.contains(baseNoiseID)).map(
      c => (c._1.replace(baseNoiseID, "").tail, c._2)
    )

    covariance.setHyperParameters(base_kernel_state)
    noise_psd.setHyperParameters(base_noise_state)

    val op_state = h.filterNot(c => c._1.contains(baseCovID) || c._1.contains(baseNoiseID))

    op_state.foreach((keyval) => operator_state += (keyval._1 -> keyval._2))

    current_state = operator_state ++
      covStEncoder(covariance.state) ++
      noiseStEncoder(noise_psd.state)

  }

  def getParams(h: Map[String, Double]): (DenseVector[Double], DenseMatrix[Double]) = {
    setState(h)

    println("Constructing Model for PSD")
    print("Dimension  = ")
    pprint.pprintln(basis.dimension)

    val dll = diffusionField(operator_state)
    val grad_dll = diffusionField.gradL.apply(operator_state)
    val lambda = lossTimeScale(operator_state)
    val q = injection_process(operator_state)

    val psi_basis = basis.operator_basis(dll, grad_dll, lambda)

    val (psi_stream, f_stream) = ghost_points.map(p => (psi_basis(p), (q(p) - lambda(p)*psd_mean)/psd_std)).unzip

    val (psi,f) = (
      DenseMatrix.vertcat(psi_stream.map(_.toDenseMatrix):_*),
      DenseVector(f_stream.toArray))

    val (no, nc) = (num_observations, num_colocation_points)

    val ones_obs = DenseVector.fill[Double](no)(1d)

    val zeros_col = DenseVector.zeros[Double](nc)

    val omega_phi = phi*phi.t

    val omega_cross = phi*psi.t

    val omega_psi = psi*psi.t

    val responses = DenseVector.vertcat(DenseVector(0d), targets.map(psd => (psd - psd_mean)/psd_std), f)

    def I(n: Int) = DenseMatrix.eye[Double](n)

    val A = DenseMatrix.vertcat(
      DenseMatrix.horzcat(DenseMatrix(0d), ones_obs.toDenseMatrix, zeros_col.toDenseMatrix),
      DenseMatrix.horzcat(ones_obs.toDenseMatrix.t, omega_phi+I(no)*regObs, omega_cross),
      DenseMatrix.horzcat(zeros_col.toDenseMatrix.t, omega_cross.t, omega_psi+(quadrature_weights*regCol))
    )

    (A\responses, psi)
  }

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    * */
  def energy(
    h: Map[String, Double],
    options: Map[String, String] = Map()): Double = try {

    val (params, psi) = getParams(h)

    val dMat = DenseMatrix.vertcat(
      DenseVector.ones[Double](num_observations).toDenseMatrix,
      phi*phi.t,
      psi*phi.t
    )

    val surrogate = dMat.t*params

    val modelVariance = norm(targets.map(psd => (psd - psd_mean)/psd_std) - surrogate)/num_observations
    print("variance = ")
    pprint.pprintln(modelVariance)

    /*
    * Construct partitioned covariance matrix
    * */

    println("Constructing partitions of covariance matrix")

    println("Partition K_uu")
    val k_uu = covariance.buildKernelMatrix(
      psd_data.map(_._1),
      num_observations).getKernelMatrix

    println("Partition K_nn")
    val noise_mat_psd = noise_psd.buildKernelMatrix(
      psd_data.map(_._1),
      num_observations).getKernelMatrix


    AbstractGPRegressionModel.logLikelihood(
      targets.map(psd => (psd - psd_mean)/psd_std) - surrogate,
      k_uu + noise_mat_psd + phi*phi.t)

  } catch {
    case _: breeze.linalg.MatrixSingularException => Double.NaN
    case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
    case _: breeze.linalg.MatrixNotSymmetricException => Double.NaN
  }

}

object SGRadialDiffusionModel {

  sealed trait QuadratureRule

  case class GaussianQuadrature(nodes: Seq[Double], weights: Seq[Double]) extends QuadratureRule {

    def scale(lower: Double, upper: Double): (Seq[Double], Seq[Double]) = {

      val sc_nodes = nodes.map(n => {
        val mid_point = (lower + upper) / 2d

        val mid_diff  = (lower + upper) / 2d

        mid_point + mid_diff*n
      })

      val sc_weights = weights.map(_*(upper - lower)/2d)

      (sc_nodes, sc_weights)
    }

    def integrate(f: (Double) => Double)(lower: Double, upper: Double): Double = {

      val (sc_nodes, sc_weights) = scale(lower, upper)

      sc_weights.zip(sc_nodes.map(f)).map(c => c._2*c._1).sum
    }
  }


  val twoPointGaussLegendre = GaussianQuadrature(
    Seq(-0.5773502692d, 0.5773502692d),
    Seq( 1d,            1d)
  )

  val threePointGaussLegendre = GaussianQuadrature(
    Seq(-0.7745966692, 0d,           0.7745966692),
    Seq( 0.5555555556, 0.8888888888, 0.5555555556)
  )


  val fourPointGaussLegendre = GaussianQuadrature(
    Seq(-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116),
    Seq( 0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451)
  )

  val fivePointGaussLegendre = GaussianQuadrature(
    Seq(-0.9061798459, -0.5384693101, 0d,           0.5384693101, 0.9061798459),
    Seq( 0.2369268851,  0.4786286705, 0.5688888888, 0.4786286705, 0.2369268851)
  )

  val sixPointGaussLegendre = GaussianQuadrature(
    Seq(-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142),
    Seq( 0.1713244924,  0.3607615730,  0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924)
  )

  val sevenPointGaussLegendre = GaussianQuadrature(
    Seq(-0.9491079123, -0.7415311856, -0.4058451514, 0d,           0.4058451514, 0.7415311856, 0.9491079123),
    Seq( 0.1294849662,  0.2797053915,  0.3818300505, 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662)
  )

  val eightPointGaussLegendre = GaussianQuadrature(
    Seq(
      -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425, 0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565
    ),
    Seq(
      0.1012285363,  0.2223810345,  0.3137066459,  0.3626837834, 0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
    )
  )

  def stateEncoder(prefix: String): Encoder[Map[String, Double], Map[String, Double]] = Encoder(
    (s: Map[String, Double]) => s.map(h => (prefix+"/"+h._1, h._2)),
    (s: Map[String, Double]) => s.map(h => (h._1.replace(prefix, "").tail, h._2))
  )

  val metaDesignMatFlow = MetaPipe((bf: Basis[(Double, Double)]) => (s: Stream[(Double, Double)]) => (
    StreamDataPipe(bf) >
      StreamDataPipe((v: DenseVector[Double]) => v.toDenseMatrix) >
      DataPipe((s: Stream[DenseMatrix[Double]]) => DenseMatrix.vertcat(s:_*)))(s)
  )

  def loadCachedResults(
    lambda_alpha: Double, lambda_beta: Double,
    lambda_a: Double, lambda_b: Double)(file: String): Stream[DenseVector[Double]] = {

    val strToVector = StreamDataPipe((p: String) => DenseVector(p.split(",").map(_.toDouble)))

    val load_results = fileToStream > strToVector

    val post_samples = load_results(".cache/"+file)

    scatter(post_samples.map(c => (c(0), c(2))))
    hold()
    scatter(Seq((math.log(math.exp(lambda_alpha)*math.pow(10d, lambda_a)), lambda_b)))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples:- "+0x03B1.toChar+" vs b")
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": b")
    unhold()



    scatter(post_samples.map(c => (c(0), c(1))))
    hold()
    scatter(Seq((math.log(math.exp(lambda_alpha)*math.pow(10d, lambda_a)), lambda_beta)))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples "+0x03B1.toChar+" vs "+0x03B2.toChar)
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": "+0x03B2.toChar)
    unhold()

    post_samples
  }


}