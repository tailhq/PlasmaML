package io.github.mandar2812.PlasmaML.dynamics.diffusion

import io.github.mandar2812.dynaml.kernels.{GenExpSpaceTimeKernel, LinearPDEKernel}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, MetaPipe}

/**
  * Radial Diffusion Kernel
  *
  * @author mandar2812 date 07/07/2017.
  * */
trait GenRadialDiffusionKernel[I] extends
  LinearPDEKernel[I] {

  override val baseKernel: GenExpSpaceTimeKernel[I]

  val diffusionField: MetaPipe[Map[String, Double], (I, Double), Double]

  val lossTimeScale: MetaPipe[Map[String, Double], (I, Double), Double]


}

class SE1dExtRadDiffusionKernel(
  sigma: Double, theta_space: Double,
  theta_time: Double, val Kp: DataPipe[Double, Double])(
  dll_params: (Double, Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double, Double)) extends
  GenRadialDiffusionKernel[Double] {

  override val baseKernel = new GenExpSpaceTimeKernel[Double](
    sigma, theta_space, theta_time)(
    sqNormDouble, sqNormDouble)

  override val diffusionField: MagTrend = new MagTrend(Kp, "dll")

  override val lossTimeScale: MagTrend = new MagTrend(Kp, "tau")

  override val baseID = "base::"+baseKernel.toString.split("\\.").last

  override def _operator_hyper_parameters = operator_hyper_parameters

  protected val operator_hyper_parameters: List[String] = {

    val dll_hyp = diffusionField.transform.keys
    val tau_hyp = lossTimeScale.transform.keys

    List(
      dll_hyp._1, dll_hyp._2, dll_hyp._3, dll_hyp._4, dll_hyp._5,
      tau_hyp._1, tau_hyp._2, tau_hyp._3, tau_hyp._4, tau_hyp._5
    )
  }

  override val hyper_parameters =
    baseKernel.hyper_parameters.map(h => baseID+"/"+h) ++
      operator_hyper_parameters

  val diffusionFieldGradL = diffusionField.gradL

  protected val gradDByLSq: MetaPipe[Map[String, Double], (Double, Double), Double] =
    MetaPipe((hyper_params: Map[String, Double]) => (x: (Double, Double)) => {
      diffusionFieldGradL(hyper_params)(x) - 2d*diffusionField(hyper_params)(x)/x._1
    })

  protected var operator_state: Map[String, Double] =
    diffusionField.transform.i(dll_params) ++ lossTimeScale.transform.i(tau_params)

  state = baseKernel.state.map(c => (baseID+"/"+c._1, c._2)) ++ operator_state


  override def invOperatorKernel = {
    val (theta_s, theta_t) = (baseKernel.state("spaceScale"), baseKernel.state("timeScale"))

    val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

    val op_state = state.filterNot(_._1.contains(baseID))

    (x: exIndexSet, x_tilda: exIndexSet) => {

      val (l,t) = x
      val (l_tilda, t_tilda) = x_tilda

      def grT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

      def grL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

      val sq = (s: Double) => s*s

      val gs_tilda = 0.5*invThetaS*sq(grL(0, 1)) - grL(0, 2)

      val tau_tilda = lossTimeScale(op_state)(x_tilda)

      val dll_tilda = diffusionField(op_state)(x_tilda)

      val alpha_tilda = gradDByLSq(op_state)(x_tilda)


      baseKernel.evaluate(x, x_tilda)*(
        0.5*invThetaS*dll_tilda*gs_tilda -
          0.5*invThetaS*alpha_tilda*grL(0, 1) +
          0.5*invThetaT*grT(0, 1) -
          tau_tilda
      )
    }
  }


  override def evaluateAt(config: Map[String, Double])(x: (Double, Double), x_tilda: (Double, Double)) = {

    val base_kernel_state = config.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, "").tail, c._2))

    val (theta_s, theta_t) = (base_kernel_state("spaceScale"), base_kernel_state("timeScale"))

    val op_state = config.filterNot(_._1.contains(baseID))

    val baseK = baseKernel.evaluateAt(base_kernel_state) _

    val (l,t) = x
    val (l_tilda, t_tilda) = x_tilda

    def grT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

    def grL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

    val sq = (s: Double) => s*s

    val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

    val (gs, gs_tilda, gs_cross) = (
      0.5*invThetaS*sq(grL(1, 0)) - grL(2, 0),
      0.5*invThetaS*sq(grL(0, 1)) - grL(0, 2),
      0.5*invThetaS*grL(1, 0)*grL(0,1) - grL(1, 1)
    )

    val (gt, gt_tilda, gt_cross) = (
      0.5*invThetaT*sq(grT(1, 0)) - grT(2, 0),
      0.5*invThetaT*sq(grT(0, 1)) - grT(0, 2),
      0.5*invThetaT*grT(1, 0)*grT(0,1) - grT(1, 1)
    )

    val (tau, tau_tilda) = (lossTimeScale(op_state)(x), lossTimeScale(op_state)(x_tilda))

    val (dll, dll_tilda) = (diffusionField(op_state)(x), diffusionField(op_state)(x_tilda))

    val (alpha, alpha_tilda) = (gradDByLSq(op_state)(x), gradDByLSq(op_state)(x_tilda))



    baseK(x, x_tilda)*(
      //Terms of nabla_ll(operator_t)
      0.5*sq(invThetaS)*dll*dll_tilda*(
        sq(grL(1,1)) - 0.5*invThetaS*grL(1,0)*grL(0,1)*grL(1,1)*(1d + grL(2,0)) + 0.5*gs*gs_tilda
        ) +
        0.25*dll*alpha_tilda*sq(invThetaS)*(2d*grL(1,0)*grL(1,1) - grL(0,1)*gs) -
        0.25*dll*invThetaS*invThetaT*grT(0, 1)*gs -
        0.5*invThetaS*dll*tau_tilda*gs +
        //Terms of nabla_l(operator_t)
        0.5*dll_tilda*alpha*sq(invThetaS)*(grL(0,1)*grL(1,1) - 0.5*gs_tilda) +
        0.5*alpha*alpha_tilda*invThetaS*gs_cross -
        0.25*alpha*invThetaS*invThetaT*grT(0, 1)*grL(1, 0) +
        0.5*invThetaS*tau_tilda*alpha*grL(1, 0) +
        //Terms nabla_t(operator_t)
        0.25*dll_tilda*invThetaS*invThetaT*gs_tilda*grT(1, 0) -
        0.25*alpha_tilda*invThetaS*invThetaT*grT(1,0)*grL(0,1) +
        0.5*invThetaT*gt_cross +
        0.5*invThetaT*tau_tilda*grT(1,0) -
        //Terms tau(operator_t)
        0.5*invThetaS*tau*gs_tilda +
        0.5*invThetaS*alpha_tilda*tau*grL(0,1) -
        0.5*invThetaT*tau*grT(0,1) +
        tau*tau_tilda
      )

  }

  //TODO: Complete implementation of gradient
  override def gradientAt(config: Map[String, Double])(x: (Double, Double), x_tilda: (Double, Double)) = {

    val base_kernel_state = config.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, "").tail, c._2))

    val (theta_s, theta_t) = (base_kernel_state("spaceScale"), base_kernel_state("timeScale"))

    val op_state = config.filterNot(_._1.contains(baseID))


    baseKernel.gradientAt(base_kernel_state)(x, x_tilda).map(c => (baseID+"/"+c._1, c._2)) ++
      operator_hyper_parameters.map(h => (h, 0d))
  }
}

class SE1dDiffusionKernel(
  sigma: Double, theta_space: Double,
  theta_time: Double, val Kp: DataPipe[Double, Double])(
  dll_params: (Double, Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double, Double)) extends
  GenRadialDiffusionKernel[Double] {


  override val baseKernel = new GenExpSpaceTimeKernel[Double](
    sigma, theta_space, theta_time)(
    sqNormDouble, sqNormDouble)

  override val diffusionField: MagTrend = new MagTrend(Kp, "dll")

  override val lossTimeScale: MagTrend = new MagTrend(Kp, "tau")

  val diffusionFieldGradL = diffusionField.gradL

  protected val gradDByLSq: MetaPipe[Map[String, Double], (Double, Double), Double] =
    MetaPipe((hyper_params: Map[String, Double]) => (x: (Double, Double)) => {
      diffusionFieldGradL(hyper_params)(x) - 2d*diffusionField(hyper_params)(x)/x._1
    })

  override val baseID = "base::"+baseKernel.toString.split("\\.").last

  override def _operator_hyper_parameters = operator_hyper_parameters

  protected val operator_hyper_parameters: List[String] = {

    val dll_hyp = diffusionField.transform.keys
    val tau_hyp = lossTimeScale.transform.keys

    List(
      dll_hyp._1, dll_hyp._2, dll_hyp._3, dll_hyp._4, dll_hyp._5,
      tau_hyp._1, tau_hyp._2, tau_hyp._3, tau_hyp._4, tau_hyp._5
    )
  }

  override val hyper_parameters =
    baseKernel.hyper_parameters.map(h => baseID+"/"+h) ++
      operator_hyper_parameters


  protected var operator_state: Map[String, Double] =
    diffusionField.transform.i(dll_params) ++ lossTimeScale.transform.i(tau_params)

  state = baseKernel.state.map(c => (baseID+"/"+c._1, c._2)) ++ operator_state

  override def invOperatorKernel = {
    val (theta_s, theta_t) = (baseKernel.state("spaceScale"), baseKernel.state("timeScale"))
    val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

    val sq = (s: Double) => s*s

    val op_state = state.filterNot(_._1.contains(baseID))

    (x: exIndexSet, x_tilda: exIndexSet) => {

      val (l,t) = x
      val (l_tilda, t_tilda) = x_tilda

      def grT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

      def grL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

      val gs_tilda = 0.5*invThetaS*sq(grL(0, 1)) - grL(0, 2)

      val tau_tilda = lossTimeScale(op_state)(x_tilda)

      val dll_tilda = diffusionField(op_state)(x_tilda)

      val alpha_tilda = gradDByLSq(op_state)(x_tilda)

      baseKernel.evaluate(x, x_tilda)*(
        0.5*invThetaS*dll_tilda*gs_tilda -
          0.5*invThetaS*alpha_tilda*grL(0, 1) +
          0.5*invThetaT*grT(0, 1)
      )/tau_tilda
    }
  }

  override def evaluateAt(config: Map[String, Double])(x: (Double, Double), x_tilda: (Double, Double)) = {

    val base_kernel_state = config.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, "").tail, c._2))

    val (theta_s, theta_t) = (base_kernel_state("spaceScale"), base_kernel_state("timeScale"))

    val op_state = config.filterNot(_._1.contains(baseID))

    val baseK = baseKernel.evaluateAt(base_kernel_state) _

    val (l,t) = x
    val (l_tilda, t_tilda) = x_tilda

    def grT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

    def grL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

    val sq = (s: Double) => s*s

    val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

    val (gs, gs_tilda, gs_cross) = (
      0.5*invThetaS*sq(grL(1, 0)) - grL(2, 0),
      0.5*invThetaS*sq(grL(0, 1)) - grL(0, 2),
      0.5*invThetaS*grL(1, 0)*grL(0,1) - grL(1, 1)
    )

    val (gt, gt_tilda, gt_cross) = (
      0.5*invThetaT*sq(grT(1, 0)) - grT(2, 0),
      0.5*invThetaT*sq(grT(0, 1)) - grT(0, 2),
      0.5*invThetaT*grT(1, 0)*grT(0,1) - grT(1, 1)
    )

    val (tau, tau_tilda) = (lossTimeScale(op_state)(x), lossTimeScale(op_state)(x_tilda))

    val (dll, dll_tilda) = (diffusionField(op_state)(x), diffusionField(op_state)(x_tilda))

    val (alpha, alpha_tilda) = (gradDByLSq(op_state)(x), gradDByLSq(op_state)(x_tilda))



    baseK(x, x_tilda)*(
      //Terms of nabla_ll(operator_t)
      0.5*sq(invThetaS)*dll*dll_tilda*(
        sq(grL(1,1)) - 0.5*invThetaS*grL(1,0)*grL(0,1)*grL(1,1)*(1d + grL(2,0)) + 0.5*gs*gs_tilda
        ) +
        0.25*dll*alpha_tilda*sq(invThetaS)*(2d*grL(1,0)*grL(1,1) - grL(0,1)*gs) -
        0.25*dll*invThetaS*invThetaT*grT(0, 1)*gs +
        //Terms of nabla_l(operator_t)
        0.5*dll_tilda*alpha*sq(invThetaS)*(grL(0,1)*grL(1,1) - 0.5*gs_tilda) +
        0.5*alpha*alpha_tilda*invThetaS*gs_cross -
        0.25*alpha*invThetaS*invThetaT*grT(0, 1)*grL(1, 0) +
        //Terms nabla_t(operator_t)
        0.25*dll_tilda*invThetaS*invThetaT*gs_tilda*grT(1, 0) -
        0.25*alpha_tilda*invThetaS*invThetaT*grT(1,0)*grL(0,1) +
        0.5*invThetaT*gt_cross
      )/(tau*tau_tilda)

  }

  //TODO: Complete implementation
  override def gradientAt(config: Map[String, Double])(x: (Double, Double), x_tilda: (Double, Double)) = {

    val base_kernel_state = config.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, "").tail, c._2))

    val (theta_s, theta_t) = (base_kernel_state("spaceScale"), base_kernel_state("timeScale"))

    val op_state = config.filterNot(_._1.contains(baseID))


    val (l,t) = x
    val (l_tilda, t_tilda) = x_tilda

    def gradDistT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

    def gradDistL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

    val (gamma, gamma_tilda) = (
      gradDistL(2, 0) - 0.5*gradDistL(1, 0)/theta_s,
      gradDistL(0, 2) - 0.5*gradDistL(0, 1)/theta_s)

    val (tau, tau_tilda) = (lossTimeScale(op_state)(x), lossTimeScale(op_state)(x_tilda))

    val (dll, dll_tilda) = (diffusionField(op_state)(x), diffusionField(op_state)(x_tilda))

    val (alpha, alpha_tilda) = (gradDByLSq(op_state)(x), gradDByLSq(op_state)(x_tilda))

    val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

    val delta = 0.25 * dll * dll_tilda * gamma * gamma_tilda * invThetaS * invThetaS +
      0.25 * dll * alpha_tilda * invThetaS * invThetaS * gradDistL(0, 1) * gamma -
      0.25 * dll * invThetaS * invThetaT * gradDistT(0, 1) * gamma +
      //Terms of nabla_l(operator_t)
      0.25 * dll_tilda * alpha * invThetaS * invThetaS * (gradDistL(1, 0) * gamma_tilda + gradDistL(1, 1)) +
      0.5 * alpha * alpha_tilda * invThetaS * (0.5 * invThetaS * gradDistL(1, 0) * gradDistL(0, 1) - gradDistL(1, 1)) -
      0.25 * alpha * invThetaS * invThetaT * gradDistT(0, 1) * gradDistL(1, 0) -
      //Terms nabla_t(operator_t)
      0.25 * dll_tilda * invThetaS * invThetaT * gamma_tilda * gradDistT(1, 0) -
      0.25 * alpha_tilda * invThetaS * invThetaT * gradDistT(1, 0) * gradDistL(0, 1) +
      0.5 * invThetaT * (0.5 * invThetaT * gradDistT(1, 0) * gradDistT(0, 1) - gradDistT(1, 1))

    baseKernel.gradientAt(base_kernel_state)(x, x_tilda).map(c => (baseID+"/"+c._1, c._2*tau*tau_tilda*delta)) ++
      operator_hyper_parameters.map(h => (h, 0d))
  }



}
