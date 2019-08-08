package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Implementation of a discrete radial diffusion system,
  * parameterised by magnetospheric processes.
  *
  * &part;&fnof;/&part;t =
  * L<sup>2</sup>&part;/&part;L(D<sub>LL</sub> &times; L<sup>-2</sup> &times; &part;&fnof;/&part;L)
  * - &lambda;(L,t) &times; &fnof;(L,t)
  * + Q(L,t)
  *
  * This class solves the radial diffusion dynamics, when supplied
  * with the magnetospheric parameters.
  *
  * @param lShellLimits The minimum and maximum value of L* i.e. the drift shell
  * @param timeLimits The minimum and maximum of the time coordinate.
  * @param nL The number of bins to divide spatial domain into.
  * @param nT The number of bins to divide temporal domain into.
  * @author mandar2812 date 30/03/2017.
  * */
class MagRadialDiffusion[T](
  diffusion: MagnetosphericProcessTrend[T],
  lossRate:  MagnetosphericProcessTrend[T],
  injection: MagnetosphericProcessTrend[T])(
  lShellLimits: (Double, Double),
  timeLimits: (Double, Double),
  nL: Int, nT: Int)
  extends Serializable {

  import MagRadialDiffusion._

  private lazy val diffusion_solver: RadialDiffusion =
    new RadialDiffusion(lShellLimits, timeLimits, nL, nT)


  lazy val stencil: (Seq[Double], Seq[Double]) = diffusion_solver.stencil

  private lazy val (lShellVec, timeVec) = stencil

  def solve(
    diffusion_params: T,
    loss_params: T,
    injection_params: T,
    f0: DenseVector[Double]): Stream[DenseVector[Double]] = {

    val (diffusionField, lossField, injectionField) = (
      diffusion(diffusion_params), lossRate(loss_params), injection(injection_params)
    )


    val diffProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => diffusionField(lShellVec(i), timeVec(j)))

    val injectionProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => injectionField(lShellVec(i), timeVec(j)))

    val lossProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => lossField(lShellVec(i), timeVec(j)))

    diffusion_solver.solve(injectionProfile, diffProfile, lossProfile)(f0)
  }

  def solve(
    diffusion_params: T,
    loss_params: T,
    injection_params: T)(
    f0: Double => Double): Stream[DenseVector[Double]] = {


    val dll = (l: Double, t: Double) => diffusion(diffusion_params)((l, t))

    val lambda = (l: Double, t: Double) => lossRate(loss_params)((l, t))

    val q = (l: Double, t: Double) => injection(injection_params)((l, t))


    diffusion_solver.solve(q, dll, lambda)(f0)
  }

  def sensitivity(
    parameter: Parameter,
    diffusion_params: T,
    loss_params: T,
    injection_params: T,
    f0: DenseVector[Double]): Map[String, Stream[DenseVector[Double]]] = parameter match {

    case Injection(keys) => {

      val (diffusionField, lossField) = (
        diffusion(diffusion_params), lossRate(loss_params)
      )

      val diffProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => diffusionField(lShellVec(i), timeVec(j)))

      val lossProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => lossField(lShellVec(i), timeVec(j)))

      val sensitivities = injection.grad.map(grad_q_i => {

        val grad_q_mat = DenseMatrix.tabulate[Double](nL+1,nT+1)(
          (i,j) => grad_q_i(injection_params)(lShellVec(i), timeVec(j)))

        val s0: DenseVector[Double] = DenseVector.zeros(lShellVec.length)


        diffusion_solver.solve(grad_q_mat, diffProfile, lossProfile)(s0)
      })

      keys.zip(sensitivities).toMap
    }

    case LossRate(keys) => {
      val (diffusionField, lossField, injectionField) = (
        diffusion(diffusion_params), lossRate(loss_params), injection(injection_params)
      )

      val diffProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => diffusionField(lShellVec(i), timeVec(j)))

      val lossProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => lossField(lShellVec(i), timeVec(j)))

      val injectionProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => injectionField(lShellVec(i), timeVec(j)))

      val psd = diffusion_solver.solve(injectionProfile, diffProfile, lossProfile)(f0)

      val psd_mat = DenseMatrix.horzcat(psd.map(_.asDenseMatrix.t):_*)

      val sensitivities = lossRate.grad.map(grad_lambda_i => {

        val grad_lambda_mat = DenseMatrix.tabulate[Double](nL+1,nT+1)(
          (i,j) => grad_lambda_i(loss_params)(lShellVec(i), timeVec(j)))

        val s0: DenseVector[Double] = DenseVector.zeros(lShellVec.length)


        diffusion_solver.solve(-grad_lambda_mat *:* psd_mat, diffProfile, lossProfile)(s0)
      })

      keys.zip(sensitivities).toMap
    }


    case DiffusionField(keys) => {
      val (diffusionField, lossField, injectionField) = (
        diffusion(diffusion_params), lossRate(loss_params), injection(injection_params)
      )

      val diffProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => diffusionField(lShellVec(i), timeVec(j)))

      val lossProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => lossField(lShellVec(i), timeVec(j)))

      val injectionProfile = DenseMatrix.tabulate[Double](nL+1,nT+1)((i,j) => injectionField(lShellVec(i), timeVec(j)))

      val psd = diffusion_solver.solve(injectionProfile, diffProfile, lossProfile)(f0)

      val sensitivities = diffusion.grad.map(grad_dll_i => {

        val s0: DenseVector[Double] = DenseVector.zeros(lShellVec.length)

        val grad_dll_i_matrix = DenseMatrix.tabulate[Double](nL + 1, nT + 1)(
          (i,j) => grad_dll_i(diffusion_params)(lShellVec(i), timeVec(j)))

        val grad_dll_params: Seq[(Seq[Seq[Double]], Seq[Seq[Double]])] = RadialDiffusion.getModelStackParams(
          lShellLimits, timeLimits, nL, nT)(
          DenseMatrix.zeros[Double](nL + 1, nT + 1),
          grad_dll_i_matrix,
          DenseMatrix.zeros[Double](nL + 1, nT + 1)).map(
          pattern => (pattern._1, pattern._2))


        val injection_mat_eff =
          DenseMatrix.horzcat(
            psd.sliding(2).toSeq.zip(grad_dll_params).map(pattern => {

              val (psd_hist, (alph, bet)) = pattern

              val alpha_mat = DenseMatrix.tabulate[Double](nL + 1, nL + 1)(
                (i, j) => if(i == j || math.abs(i - j) == 1) alph(i)(j - i + 1) else 0d
              )

              val beta_mat = DenseMatrix.tabulate[Double](nL + 1, nL + 1)(
                (i, j) => if(i == j || math.abs(i - j) == 1) bet(i)(j - i + 1) else 0d
              )

              val invT = DenseMatrix.eye[Double](nL + 1) * (nT/(timeLimits._2 - timeLimits._1))

              val alpha_mat_adj = alpha_mat - invT

              val beta_mat_adj  = beta_mat  - invT

              alpha_mat_adj*psd_hist.head - beta_mat_adj*psd_hist.last
            }).map(_.toDenseMatrix.t):_*
          )


        diffusion_solver.solve(
          DenseMatrix.horzcat(injection_mat_eff, DenseMatrix.zeros[Double](nL + 1, 1)),
          diffProfile, lossProfile)(s0)
      })

      keys.zip(sensitivities).toMap
    }

    case _ => Map()
  }

  def sensitivity(
    parameter: Parameter)(
    diffusion_params: T,
    loss_params: T,
    injection_params: T)(
    f0: Double => Double): Map[String, Stream[DenseVector[Double]]] =
    sensitivity(
      parameter, diffusion_params, loss_params, injection_params,
      DenseVector(lShellVec.map(f0).toArray)
    )

}


object MagRadialDiffusion {

  sealed trait Parameter

  case class Injection(keys: Seq[String]) extends Parameter

  case class LossRate(keys: Seq[String]) extends Parameter

  case class DiffusionField(keys: Seq[String]) extends Parameter

  def apply[T](
    diffusion: MagnetosphericProcessTrend[T],
    lossRate: MagnetosphericProcessTrend[T],
    injection: MagnetosphericProcessTrend[T])(
    lShellLimits: (Double, Double),
    timeLimits: (Double, Double),
    nL: Int, nT: Int): MagRadialDiffusion[T] =
    new MagRadialDiffusion(
      diffusion, lossRate, injection)(
      lShellLimits, timeLimits, nL, nT)
}