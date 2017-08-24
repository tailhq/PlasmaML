package io.github.mandar2812.PlasmaML.dynamics.diffusion

import scala.reflect.ClassTag
import spire.implicits._
import breeze.linalg.{DenseMatrix, DenseVector, norm}
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.analysis._
import io.github.mandar2812.dynaml.kernels.{GenExpSpaceTimeKernel, LinearPDEKernel, LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.gp._
import io.github.mandar2812.dynaml.optimization.{GloballyOptimizable, RegularizedLSSolver}
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import io.github.mandar2812.dynaml.probability.distributions.MVGaussian
import org.apache.log4j.Logger


/**
  * <h3>Gaussian Process Radial Diffusion Model</h3>
  *
  * Implementation of a space-time magnetospheric radial diffusion system
  * as a gaussian process regression model.
  *
  * Can be used for likelihood based inverse parameter estimation or
  * forward predictive/emulation problems.
  * 
  * @tparam K A subtype/implementation of [[GenRadialDiffusionKernel]]
  * @tparam I The type of the space variable.
  *
  * @param cov The covariance structure of the process
  * @param n The measurement noise inherent in the recorded target values.
  * @param data Measurements of the field at select space-time locations.
  * @param basisFunc A transformation which takes as input space time coordinates
  *                  and outputs the basis function representation.
  * @param basis_param_prior The prior probabililty distribution over basis function
  *                          coefficients.
  * @author mandar2812 date 2017/08/13
  * */
class GPRadialDiffusionModel[I: ClassTag, K <: GenRadialDiffusionKernel[I]](
  cov: K, n: LocalScalarKernel[(I, Double)],
  data: Stream[((I, Double), Double)], num: Int,
  basisFunc: DataPipe[(I, Double), DenseVector[Double]],
  basis_param_prior: MultGaussianRV) extends GPBasisFuncRegressionModel[
  Stream[((I, Double), Double)], (I, Double)](
  cov, n, data, num, basisFunc, basis_param_prior) {


  override protected def getCrossKernelMatrix[U <: Seq[(I, Double)]](test: U) =
    SVMKernel.crossPartitonedKernelMatrix(
      trainingData, test,
      _blockSize, _blockSize,
      cov.invOperatorKernel)

  override protected def getTestKernelMatrix[U <: Seq[(I, Double)]](test: U) =
    SVMKernel.buildPartitionedKernelMatrix(
      test, test.length.toLong,
      _blockSize, _blockSize,
      cov.baseKernel.evaluate)


  override def dataAsSeq(data: Stream[((I, Double), Double)]): Seq[((I, Double), Double)] = data


}

/**
  * Inverse inference over plasma radial diffusion parameters.
  * */
class InverseRadialDiffusion[K <: GenRadialDiffusionKernel[Double]](
  val covariance: K, val noise_psd: LocalScalarKernel[(Double, Double)],
  val psd_data: Stream[((Double, Double), Double)],
  val injection_data: Stream[((Double, Double), Double)],
  val basis: PSDRadialBasis) extends GloballyOptimizable {

  protected val logger: Logger = Logger.getLogger(this.getClass)

  val psd_data_size = psd_data.length

  val psd_mean: Double = psd_data.map(_._2).sum/psd_data_size

  private lazy val targets = DenseVector(psd_data.map(_._2).toArray)

  val metaDesignMatFlow = MetaPipe((bf: Basis[(Double, Double)]) => (s: Stream[(Double, Double)]) => (
    StreamDataPipe(bf) >
      StreamDataPipe((v: DenseVector[Double]) => v.toDenseMatrix) >
      DataPipe((s: Stream[DenseMatrix[Double]]) => DenseMatrix.vertcat(s:_*)))(s)
  )

  private val designMatrixFlow = metaDesignMatFlow(basis)

  //val injection_data: Stream[((Double, Double), Double)] = basis._centers.toStream.map(p => (p, 0d))

  lazy val injection = DenseVector(injection_data.map(_._2).toArray)

  lazy val phi = designMatrixFlow(psd_data.map(_._1))

  private lazy val (aMat, b) = psd_data.map(p => {
    val ph = basis(p._1)
    val y = p._2
    (ph*ph.t, ph*y)
  }).reduceLeft((x, y) => (x._1+y._1, x._2+y._2))


  private val grid_centers = combine(Seq(basis.lSeq, basis.tSeq)).map(s => (s.head, s.last))

  private val grid_scales = combine(Seq(basis.scalesL, basis.scalesT)).map(s => (s.head, s.last))

  /**
    * Stores the names of the hyper-parameters
    * */
  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noise_psd.hyper_parameters


  /**
    * A Map which stores the current state of the system.
    * */
  override protected var current_state: Map[String, Double] =
    covariance.state ++ noise_psd.state


  var reg: Double = 0.01


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

    covariance.setHyperParameters(h)
    noise_psd.setHyperParameters(h)

    val op_state = covariance.state.filterNot(_._1.contains(covariance.baseID))

    logger.info("Constructing Radial Basis Model for PSD")
    logger.info("Dimension (l*t): "+basis.dimensionL+"*"+basis.dimensionT+" = "+basis.dimension)


    val g_basis = Basis((x: (Double, Double)) => {

      val (l, t) = x

      val dll = covariance.diffusionField(op_state)(x)
      val alpha = covariance.gradDByLSq(op_state)(x)
      val lambda = covariance.lossTimeScale(op_state)(x)

      DenseVector(
        grid_centers.zip(grid_scales).map(c => {
          val ((l_tilda, t_tilda), (theta_s, theta_t)) = c

          def grT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

          def grL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

          val sq = (s: Double) => s*s

          val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

          val gs = 0.5*invThetaS*sq(grL(1, 0)) - grL(2, 0)

          0.5*invThetaS*dll*gs - 0.5*invThetaS*alpha*grL(1, 0) + 0.5*invThetaT*grT(1, 0) - lambda
        }).toArray)
    })

    //val g = metaDesignMatFlow(g_basis)(injection_data.map(_._1))

    val (bMat, c) = injection_data.map(p => {
      val ph = basis(p._1) *:* g_basis(p._1)
      val y = p._2
      (ph*ph.t, ph*y)
    }).reduceLeft((x, y) => (x._1+y._1, x._2+y._2))

    /*val smoother = DenseMatrix.tabulate[Double](basis.dimension, basis.dimension)((i,j) => {
      if(i != j) 0.0 else if(i == 0) reg else 1.0
    })*/

    val params = (aMat + bMat*reg)\(b + c)

    /*
    * Construct partitioned covariance matrix
    * */

    logger.info("Constructing partitions of covariance matrix")

    logger.info("Partition K_uu")
    val k_uu = covariance.baseKernel.buildKernelMatrix(
      psd_data.map(_._1),
      psd_data_size).getKernelMatrix

    val noise_mat_psd = noise_psd.buildKernelMatrix(
      psd_data.map(_._1),
      psd_data_size).getKernelMatrix

    val mean = phi*params

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

