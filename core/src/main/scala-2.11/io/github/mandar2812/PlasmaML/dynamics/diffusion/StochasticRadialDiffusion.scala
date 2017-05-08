package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.bayes.CoRegGPPrior
import io.github.mandar2812.dynaml.probability.MatrixNormalRV

/**
  * Created by mandar on 04/05/2017.
  * */
class StochasticRadialDiffusion[ParamsQ, ParamsD](
  psdCovarianceL: StochasticRadialDiffusion.Kernel,
  psdCovarianceT: StochasticRadialDiffusion.Kernel,
  injectionProcess: StochasticRadialDiffusion.LatentProcess[ParamsQ],
  diffusionProcess: StochasticRadialDiffusion.LatentProcess[ParamsD],
  linearDecay: Boolean = false) {

  val forwardSolver = (lDomain: (Double, Double), nL: Int, timeDomain: (Double, Double), nT: Int) =>
    RadialDiffusion(lDomain, timeDomain, nL, nT, linearDecay)

  def priorDistribution(
    lDomain: (Double, Double), nL: Int,
    timeDomain: (Double, Double), nT: Int)(
    f0: DenseVector[Double]): MatrixNormalRV = {

    val radialSolver = forwardSolver(lDomain, nL, timeDomain, nT)

    val l_values = radialSolver.lShellVec.toArray.toSeq
    val t_values = radialSolver.timeVec.toArray.toSeq

    val dll_profile = diffusionProcess.priorDistribution(l_values, t_values).draw
    val q_profile = injectionProcess.priorDistribution(l_values, t_values).draw

    val solution = radialSolver.solve(q_profile, dll_profile, DenseMatrix.zeros[Double](nL+1, nT))(f0)

    val m = DenseMatrix.horzcat(solution.tail.map(_.asDenseMatrix.t):_*)

    val u = psdCovarianceL.buildKernelMatrix(l_values, l_values.length).getKernelMatrix()
    val v = psdCovarianceT.buildKernelMatrix(t_values.tail, t_values.tail.length).getKernelMatrix()

    MatrixNormalRV(m, u, v)
  }


}

object StochasticRadialDiffusion {

  type LatentProcess[MP] = CoRegGPPrior[Double, Double, MP]
  type Kernel = LocalScalarKernel[Double]

  def apply[MPD, MPQ](
    psdCovarianceL: Kernel, psdCovarianceT: Kernel,
    injectionProcess: LatentProcess[MPQ], diffusionProcess: LatentProcess[MPD],
    linearDecay: Boolean = false) =
    new StochasticRadialDiffusion(
      psdCovarianceL, psdCovarianceT,
      injectionProcess, diffusionProcess)

}