package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, SVMKernel}
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
  *
  * @param noise_psd A kernel function representing the measurement noise of the
  *                  Phase Space Density at a pair of space time locations.
  *
  * @param psd_data A Stream of space time locations and measured PSD values.
  *
  * @param colocation_points A collection of "ghost" points on which Particle diffusion is computed
  *                     and its dependence on PSD is enforced with square loss.
  *
  *
  * */
class KernelRadialDiffusionModel(
  val Kp: DataPipe[Double, Double],
  dll_params: (Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double),
  q_params: (Double, Double, Double, Double))(
  sigma: Double, thetaS: Double, thetaT: Double,
  val noise_psd: LocalScalarKernel[(Double, Double)],
  val noise_injection: LocalScalarKernel[(Double, Double)],
  val psd_data: Stream[((Double, Double), Double)],
  val colocation_points: Stream[(Double, Double)]) extends
  GloballyOptimizable {

  protected val logger: Logger = Logger.getLogger(this.getClass)

  val baseNoiseID: String = "base_noise::"+noise_psd.toString.split("\\.").last

  val baseInjNoiseID: String = "base_inj_noise::"+noise_injection.toString.split("\\.").last

  val injection_process: MagTrend = new MagTrend(Kp, "Q")

  val num_observations: Int = psd_data.length

  val num_colocation_points: Int = colocation_points.length

  val psd_mean: Double = psd_data.map(_._2).sum/num_observations

  private lazy val targets = DenseVector(psd_data.map(_._2).toArray)

  private val (noiseStEncoder, injNoiseStEncoder) = (
    BasisFuncRadialDiffusionModel.stateEncoder(baseNoiseID),
    BasisFuncRadialDiffusionModel.stateEncoder(baseInjNoiseID)
  )

  val covariance = new SE1dExtRadDiffusionKernel(sigma, thetaS, thetaT, Kp)(dll_params, tau_params, "L2", "L1")

  protected val operator_hyper_parameters: List[String] = {

    val dll_hyp = covariance.diffusionField.transform.keys
    val tau_hyp = covariance.lossTimeScale.transform.keys
    val q_hyp = injection_process.transform.keys

    List(
      dll_hyp._1, dll_hyp._2, dll_hyp._3, dll_hyp._4,
      tau_hyp._1, tau_hyp._2, tau_hyp._3, tau_hyp._4,
      q_hyp._1, q_hyp._2, q_hyp._3, q_hyp._4
    )
  }

  def _operator_hyper_parameters: List[String] = operator_hyper_parameters

  /**
    * Stores the value of the operator parameters
    * as a [[Map]].
    * */
  protected var operator_state: Map[String, Double] = {
    val dll_hyp = covariance.diffusionField.transform.keys
    val tau_hyp = covariance.lossTimeScale.transform.keys
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
    covariance._base_hyper_parameters ++
      noise_psd.hyper_parameters.map(h => baseNoiseID+"/"+h) ++
      noise_injection.hyper_parameters.map(h => baseInjNoiseID+"/"+h) ++
      operator_hyper_parameters


  /**
    * A Map which stores the current state of the system.
    * */
  override protected var current_state: Map[String, Double] =
    covariance._base_state ++
      noiseStEncoder(noise_psd.state) ++
      injNoiseStEncoder(noise_injection.state) ++
      operator_state


  var blocked_hyper_parameters: List[String] =
    covariance.blocked_hyper_parameters ++
      noise_psd.blocked_hyper_parameters.map(h => baseNoiseID+"/"+h) ++
      noise_injection.blocked_hyper_parameters.map(h => baseInjNoiseID+"/"+h)

  def block(hyp: String*): Unit = {

    val proc_cov_hyp = hyp.filter(
      h => h.contains(covariance.baseID) || h.contains("tau") || h.contains("dll"))

    val proc_noise_hyp = hyp.filter(_.contains(baseNoiseID)).map(h => h.replace(baseNoiseID, "").tail)

    val proc_inj_noise_hyp = hyp.filter(
      _.contains(baseInjNoiseID)).map(
      h => h.replace(baseInjNoiseID, "").tail)

    covariance.block(proc_cov_hyp:_*)
    noise_psd.block(proc_noise_hyp:_*)
    noise_injection.block(proc_inj_noise_hyp:_*)
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
      c => c.contains(covariance.baseID) || c.contains("tau") || c.contains("dll")
    )


    val base_noise_state = h.filterKeys(
      _.contains(baseNoiseID)).map(
      c => (c._1.replace(baseNoiseID, "").tail, c._2)
    )

    val base_inj_noise_state = h.filterKeys(
      _.contains(baseInjNoiseID)).map(
      c => (c._1.replace(baseInjNoiseID, "").tail, c._2)
    )

    covariance.setHyperParameters(base_kernel_state)
    noise_psd.setHyperParameters(base_noise_state)
    noise_injection.setHyperParameters(base_inj_noise_state)


    val op_state = h.filterNot(
      c => c._1.contains(covariance.baseID) || c._1.contains(baseNoiseID) || c._1.contains(baseInjNoiseID))

    op_state.foreach((keyval) => operator_state += (keyval._1 -> keyval._2))

    current_state = operator_state ++
      covariance._base_state ++
      noiseStEncoder(noise_psd.state)

  }


  def getGalerkinParams(h: Map[String, Double]): (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    setState(h)

    logger.info("Constructing Kernel Primal-Dual Model for PSD")

    val q = injection_process(operator_state)

    val (no, nc) = (num_observations, num_colocation_points)

    val ones_obs = DenseVector.fill[Double](no)(1d)

    val zeros_col = DenseVector.zeros[Double](nc)

    val omega_phi = covariance.baseKernel.buildKernelMatrix(psd_data.map(_._1), no).getKernelMatrix()

    val omega_cross = SVMKernel.crossKernelMatrix(psd_data.map(_._1), colocation_points, covariance.invOperatorKernel)

    val omega_psi = covariance.buildKernelMatrix(colocation_points, nc).getKernelMatrix()

    val responses = DenseVector.vertcat(
      DenseVector(0d), targets,
      DenseVector(colocation_points.map(p => q(p)).toArray)
    )

    def I(n: Int) = DenseMatrix.eye[Double](n)

    val A = DenseMatrix.vertcat(
      DenseMatrix.horzcat(DenseMatrix(0d), ones_obs.toDenseMatrix, zeros_col.toDenseMatrix),
      DenseMatrix.horzcat(ones_obs.toDenseMatrix.t, omega_phi+I(no)*noise_psd.state("noiseLevel"), omega_cross),
      DenseMatrix.horzcat(zeros_col.toDenseMatrix.t, omega_cross.t, omega_psi+I(nc)*noise_injection.state("noiseLevel"))
    )

    (A\responses, omega_phi, omega_cross)
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

    val (params, omega, omega_cross) = getGalerkinParams(h)

    val dMat = DenseMatrix.vertcat(
      DenseVector.ones[Double](num_observations).toDenseMatrix,
      omega,
      omega_cross.t
    )

    val mean = dMat.t*params

    val modelVariance = norm(targets - mean)/targets.length
    logger.info("variance: "+modelVariance)

    /*
    * Construct partitioned covariance matrix
    * */

    logger.info("Constructing partitions of covariance matrix")

    logger.info("Partition K_uu")
    val k_uu = covariance.baseKernel.buildKernelMatrix(
      psd_data.map(_._1),
      num_observations).getKernelMatrix

    logger.info("Partition K_nn")
    val noise_mat_psd = noise_psd.buildKernelMatrix(
      psd_data.map(_._1),
      num_observations).getKernelMatrix


    AbstractGPRegressionModel.logLikelihood(targets - mean, k_uu + noise_mat_psd )
  } catch {
    case _: breeze.linalg.MatrixSingularException => Double.NaN
    case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
    case _: breeze.linalg.MatrixNotSymmetricException => Double.NaN
  }

}
