package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import io.github.mandar2812.dynaml.algebra.{PartitionedVector, PartitionedPSDMatrix}
import io.github.mandar2812.dynaml.models.gp.GPBasisFuncRegressionModel
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, GenExpSpaceTimeKernel, LinearPDEKernel, SVMKernel}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, MetaPipe}


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
abstract class GPRadialDiffusionModel[I, K <: GenRadialDiffusionKernel[I]](
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




}

