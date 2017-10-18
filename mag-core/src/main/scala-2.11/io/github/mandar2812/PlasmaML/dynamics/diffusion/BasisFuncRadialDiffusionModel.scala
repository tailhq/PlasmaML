package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes._
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
class BasisFuncRadialDiffusionModel(
  val Kp: DataPipe[Double, Double],
  dll_params: (Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double),
  q_params: (Double, Double, Double, Double))(
  val covariance: LocalScalarKernel[(Double, Double)],
  val noise_psd: DiracTuple2Kernel,
  val psd_data: Stream[((Double, Double), Double)],
  val ghost_points: Stream[(Double, Double)],
  val basis: PSDBasis) extends GloballyOptimizable {

  protected val logger: Logger = Logger.getLogger(this.getClass)

  val baseCovID: String = "base::"+covariance.toString.split("\\.").last

  val baseNoiseID: String = "base_noise::"+noise_psd.toString.split("\\.").last

  val diffusionField: MagTrend = new MagTrend(Kp, "dll")

  val lossTimeScale: MagTrend = new MagTrend(Kp, "tau")

  val injection_process: MagTrend = new MagTrend(Kp, "Q")

  val num_observations: Int = psd_data.length

  val num_colocation_points: Int = ghost_points.length

  val psd_mean: Double = psd_data.map(_._2).sum/num_observations

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

    val proc_cov_hyp = blocked_cov_hyp.filter(_.contains(baseCovID)).map(h => h.replace(baseCovID, "").tail)
    val proc_noise_hyp = blocked_cov_hyp.filter(_.contains(baseNoiseID)).map(h => h.replace(baseNoiseID, "").tail)


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

  def getBasisParams(h: Map[String, Double]): DenseVector[Double] = {
    setState(h)

    logger.info("Constructing Radial Basis Model for PSD")
    logger.info("Dimension (l*t): "+basis.dimensionL+"*"+basis.dimensionT+" = "+basis.dimension)

    val dll = diffusionField(operator_state)
    val grad_dll = diffusionField.gradL.apply(operator_state)
    val lambda = lossTimeScale(operator_state)
    val q = injection_process(operator_state)

    val g_basis = basis.operator_basis(dll, grad_dll, lambda)

    val (bMat, c) = ghost_points.map(p => {
      val ph = basis(p) *:* g_basis(p)
      val y: Double = q(p)
      (ph*ph.t, ph*y)
    }).reduceLeft((x, y) => (x._1+y._1, x._2+y._2))

    lazy val ss = aMat + bMat*reg

    ss\(b + c)
  }


  def getGalerkinParams(h: Map[String, Double]): (DenseVector[Double], DenseMatrix[Double]) = {
    setState(h)

    logger.info("Constructing Radial Basis Model for PSD")
    logger.info("Dimension (l*t): "+basis.dimensionL+"*"+basis.dimensionT+" = "+basis.dimension)

    val dll = diffusionField(operator_state)
    val grad_dll = diffusionField.gradL.apply(operator_state)
    val lambda = lossTimeScale(operator_state)
    val q = injection_process(operator_state)

    val g_basis = basis.operator_basis(dll, grad_dll, lambda)

    val (psi_stream, f_stream) = ghost_points.map(p => {
      val ph = basis(p) *:* g_basis(p)
      val y: Double = q(p)
      (ph, y)
    }).unzip

    val (psi,f) = (
      DenseMatrix.vertcat(psi_stream.map(_.toDenseMatrix):_*),
      DenseVector(f_stream.toArray))

    val (no, nc) = (num_observations, num_colocation_points)

    val ones_obs = DenseVector.fill[Double](no)(1d)

    val zeros_col = DenseVector.zeros[Double](nc)

    val omega_phi = phi*phi.t

    val omega_cross = phi*psi.t

    val omega_psi = psi*psi.t

    val responses = DenseVector.vertcat(DenseVector(0d), targets, f)

    def I(n: Int) = DenseMatrix.eye[Double](n)

    val A = DenseMatrix.vertcat(
      DenseMatrix.horzcat(DenseMatrix(0d), ones_obs.toDenseMatrix, zeros_col.toDenseMatrix),
      DenseMatrix.horzcat(ones_obs.toDenseMatrix.t, omega_phi+I(no)*regObs, omega_cross),
      DenseMatrix.horzcat(zeros_col.toDenseMatrix.t, omega_cross.t, omega_psi+I(nc)*regCol)
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

    val (params, psi) = getGalerkinParams(h)

    val dMat = DenseMatrix.vertcat(
      DenseVector.ones[Double](num_observations).toDenseMatrix,
      phi*phi.t,
      psi*phi.t
    )

    val mean = dMat.t*params

    val modelVariance = norm(targets - mean)/targets.length
    logger.info("variance: "+modelVariance)

    /*
    * Construct partitioned covariance matrix
    * */

    logger.info("Constructing partitions of covariance matrix")

    logger.info("Partition K_uu")
    val k_uu = covariance.buildKernelMatrix(
      psd_data.map(_._1),
      num_observations).getKernelMatrix

    logger.info("Partition K_nn")
    val noise_mat_psd = noise_psd.buildKernelMatrix(
      psd_data.map(_._1),
      num_observations).getKernelMatrix


    AbstractGPRegressionModel.logLikelihood(targets - mean, k_uu + noise_mat_psd)
  } catch {
    case _: breeze.linalg.MatrixSingularException => Double.NaN
    case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
    case _: breeze.linalg.MatrixNotSymmetricException => Double.NaN
  }

}

object BasisFuncRadialDiffusionModel {

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