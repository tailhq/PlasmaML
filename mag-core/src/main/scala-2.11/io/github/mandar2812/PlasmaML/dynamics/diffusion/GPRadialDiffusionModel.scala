package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import ammonite.ops._
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.distributions.MVGaussian
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
  * @param injection_data A Stream of space time locations and measured particle
  *                       injection rates.
  *
  * @param basis A basis function expansion for the PSD, as an instance
  *              of [[PSDBasis]].
  * */
class GPRadialDiffusionModel(
  val Kp: DataPipe[Double, Double],
  dll_params: (Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double))(
  val covariance: LocalScalarKernel[(Double, Double)],
  val noise_psd: LocalScalarKernel[(Double, Double)],
  val psd_data: Stream[((Double, Double), Double)],
  val injection_data: Stream[((Double, Double), Double)],
  val basis: PSDBasis) extends GloballyOptimizable {

  protected val logger: Logger = Logger.getLogger(this.getClass)

  val baseCovID: String = "base::"+covariance.toString.split("\\.").last

  val baseNoiseID: String = "base_noise::"+noise_psd.toString.split("\\.").last

  val diffusionField: MagTrend = new MagTrend(Kp, "dll")

  val lossTimeScale: MagTrend = new MagTrend(Kp, "tau")

  val psd_data_size: Int = psd_data.length

  val psd_mean: Double = psd_data.map(_._2).sum/psd_data_size

  private lazy val targets = DenseVector(psd_data.map(_._2).toArray)

  private val (covStEncoder, noiseStEncoder) = (
    GPRadialDiffusionModel.stateEncoder(baseCovID),
    GPRadialDiffusionModel.stateEncoder(baseNoiseID)
  )

  private val designMatrixFlow = GPRadialDiffusionModel.metaDesignMatFlow(basis)

  lazy val injection = DenseVector(injection_data.map(_._2).toArray)

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

    List(
      dll_hyp._1, dll_hyp._2, dll_hyp._3, dll_hyp._4,
      tau_hyp._1, tau_hyp._2, tau_hyp._3, tau_hyp._4
    )
  }

  /**
    * Stores the value of the operator parameters
    * as a [[Map]].
    * */
  protected var operator_state: Map[String, Double] = {
    val dll_hyp = diffusionField.transform.keys
    val tau_hyp = lossTimeScale.transform.keys

    Map(
      dll_hyp._1 -> dll_params._1, dll_hyp._2 -> dll_params._2,
      dll_hyp._3 -> dll_params._3, dll_hyp._4 -> dll_params._4,
      tau_hyp._1 -> tau_params._1, tau_hyp._2 -> tau_params._2,
      tau_hyp._3 -> tau_params._3, tau_hyp._4 -> tau_params._4
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

  var reg: Double = 0.01

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

    val base_kernel_state = h.filterKeys(_.contains(baseCovID)).map(c => (c._1.replace(baseCovID, "").tail, c._2))
    val base_noise_state = h.filterKeys(_.contains(baseNoiseID)).map(c => (c._1.replace(baseNoiseID, "").tail, c._2))

    covariance.setHyperParameters(base_kernel_state)
    noise_psd.setHyperParameters(base_noise_state)

    val op_state = h.filterNot(c => c._1.contains(baseCovID) || c._1.contains(baseNoiseID))

    op_state.foreach((keyval) => operator_state += (keyval._1 -> keyval._2))

    current_state = operator_state ++
      covStEncoder(covariance.state) ++
      noiseStEncoder(noise_psd.state)

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
    options: Map[String, String] = Map()): Double = {

    setState(h)

    logger.info("Constructing Radial Basis Model for PSD")
    logger.info("Dimension (l*t): "+basis.dimensionL+"*"+basis.dimensionT+" = "+basis.dimension)


    val dll = diffusionField(operator_state)
    val grad_dll = diffusionField.gradL(operator_state)
    val lambda = lossTimeScale(operator_state)

    val g_basis = basis.operator_basis(dll, grad_dll, lambda)

    val (bMat, c) = injection_data.map(p => {
      val ph = basis(p._1) *:* g_basis(p._1)
      val y = p._2
      (ph*ph.t, ph*y)
    }).reduceLeft((x, y) => (x._1+y._1, x._2+y._2))

    lazy val ss = aMat + bMat*reg

    val params = ss\(b + c)

    val mean = phi*params

    /*
    * Construct partitioned covariance matrix
    * */

    logger.info("Constructing partitions of covariance matrix")

    logger.info("Partition K_uu")
    val k_uu = covariance.buildKernelMatrix(
      psd_data.map(_._1),
      psd_data_size).getKernelMatrix


    /*logger.info("Partition K_phi")
    lazy val k_phi = phi*(ss\phi.t)
    logger.info("Volume of K_phi = "+det(k_phi))*/

    logger.info("Partition K_nn")
    val noise_mat_psd = noise_psd.buildKernelMatrix(
      psd_data.map(_._1),
      psd_data_size).getKernelMatrix

    logger.info("err: "+norm(targets - mean)/targets.length)

    val gaussian = MVGaussian(mean, k_uu + noise_mat_psd)

    try {
      -1d*gaussian.logPdf(targets)
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }
  }

}

object GPRadialDiffusionModel {

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
    lambda_a: Double, lambda_b: Double)(file: String): Unit = {

    val strToVector = StreamDataPipe((p: String) => DenseVector(p.split(",").map(_.toDouble)))

    val load_results = fileToStream > strToVector

    val post_samples = load_results(".cache/"+file)

    scatter(post_samples.map(c => (c(0), c(2))))
    hold()
    scatter(Seq((lambda_alpha*math.pow(10d, lambda_a), lambda_b)))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples:- "+0x03B1.toChar+" vs b")
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": b")
    unhold()



    scatter(post_samples.map(c => (c(0), c(1))))
    hold()
    scatter(Seq((lambda_alpha*math.pow(10d, lambda_a), lambda_beta)))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples "+0x03B1.toChar+" vs "+0x03B2.toChar)
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": "+0x03B2.toChar)
    unhold()

  }


}