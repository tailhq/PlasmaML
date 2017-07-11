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

  /*override val baseKernel: GenExpSpaceTimeKernel[I] =
    new GenExpSpaceTimeKernel[I](sigma, theta_space, theta_time)(ds, dt)*/

  val diffusionField: MetaPipe[Map[String, Double], (I, Double), Double]

  val diffusionFieldGradL: MetaPipe[Map[String, Double], (I, Double), Double]

  protected val gradDByLSq: MetaPipe[Map[String, Double], (I, Double), Double]

  val lossTimeScale: MetaPipe[Map[String, Double], (I, Double), Double]


}

class SE1dDiffusionKernel(
  sigma: Double, theta_space: Double,
  theta_time: Double, val Kp: DataPipe[Double, Double])(
  dll_params: (Double, Double, Double, Double),
  tau_params: (Double, Double, Double, Double)) extends
  GenRadialDiffusionKernel[Double] {


  override val baseKernel = new GenExpSpaceTimeKernel[Double](
    sigma, theta_space, theta_time)(
    sqNormDouble, sqNormDouble)

  override val diffusionField: MagTrend = new MagTrend(Kp, "dll")

  override val lossTimeScale: MagTrend = new MagTrend(Kp, "tau")

  override val diffusionFieldGradL = MetaPipe((hyper_params: Map[String, Double]) => (x: (Double, Double)) => {
    val (_, beta, _, _) = diffusionField.transform(hyper_params)
    val l = x._1
    diffusionField(hyper_params)(x)*beta/l
  })

  override protected val gradDByLSq: MetaPipe[Map[String, Double], (Double, Double), Double] =
    MetaPipe((hyper_params: Map[String, Double]) => (x: (Double, Double)) => {
      diffusionFieldGradL(hyper_params)(x) - 2d*diffusionField(hyper_params)(x)/x._1
    })

  override val baseID = "base::"+baseKernel.toString.split("\\.").last

  override def _operator_hyper_parameters = operator_hyper_parameters

  protected val operator_hyper_parameters: List[String] = {

    val dll_hyp = diffusionField.transform.keys
    val tau_hyp = lossTimeScale.transform.keys

    List(
      dll_hyp._1, dll_hyp._2, dll_hyp._3, dll_hyp._4,
      tau_hyp._1, tau_hyp._2, tau_hyp._3, tau_hyp._4
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

    (x: exIndexSet, x_tilda: exIndexSet) => {

      val (l,t) = x
      val (l_tilda, t_tilda) = x_tilda

      def gradDistT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

      def gradDistL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

      val tau = lossTimeScale(operator_state)

      val dll = diffusionField(operator_state)

      val alpha = gradDByLSq(operator_state)

      baseKernel.evaluate(x, x_tilda)*tau(x_tilda)*0.5*(
        gradDistT(0, 1)/theta_t -
          dll(x_tilda)*(gradDistL(0, 2) - 0.5*gradDistL(0, 1)/theta_s)/theta_s -
          alpha(x_tilda)*gradDistL(0, 1)/theta_s
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

    def gradDistT(i: Int, j: Int) = gradSqNormDouble(i, j)(t, t_tilda)

    def gradDistL(i: Int, j: Int) = gradSqNormDouble(i, j)(l, l_tilda)

    val (gamma, gamma_tilda) = (
      gradDistL(2, 0) - 0.5*gradDistL(1, 0)/theta_s,
      gradDistL(0, 2) - 0.5*gradDistL(0, 1)/theta_s)

    val (tau, tau_tilda) = (lossTimeScale(op_state)(x), lossTimeScale(op_state)(x_tilda))

    val (dll, dll_tilda) = (diffusionField(op_state)(x), diffusionField(op_state)(x_tilda))

    val (alpha, alpha_tilda) = (gradDByLSq(op_state)(x), gradDByLSq(op_state)(x_tilda))

    val (invThetaS, invThetaT) = (1/theta_s, 1/theta_t)

    baseK(x, x_tilda)*tau*tau_tilda*(
      //Terms of nabla_ll(operator_t)
      0.25*dll*dll_tilda*gamma*gamma_tilda*invThetaS*invThetaS +
        0.25*dll*alpha_tilda*invThetaS*invThetaS*gradDistL(0, 1)*gamma -
        0.25*dll*invThetaS*invThetaT*gradDistT(0, 1)*gamma +
        //Terms of nabla_l(operator_t)
        0.25*dll_tilda*alpha*invThetaS*invThetaS*(gradDistL(1,0)*gamma_tilda + gradDistL(1,1)) +
        0.5*alpha*alpha_tilda*invThetaS*(0.5*invThetaS*gradDistL(1,0)*gradDistL(0,1) - gradDistL(1,1)) -
        0.25*alpha*invThetaS*invThetaT*gradDistT(0, 1)*gradDistL(1, 0) -
        //Terms nabla_t(operator_t)
        0.25*dll_tilda*invThetaS*invThetaT*gamma_tilda*gradDistT(1, 0) -
        0.25*alpha_tilda*invThetaS*invThetaT*gradDistT(1,0)*gradDistL(0,1) +
        0.5*invThetaT*(0.5*invThetaT*gradDistT(1, 0)*gradDistT(0, 1) - gradDistT(1, 1))
      )

  }

  //TODO: Complete implementation
  override def gradientAt(config: Map[String, Double])(x: (Double, Double), x_tilda: (Double, Double)) = {

    val base_kernel_state = config.filterKeys(_.contains(baseID)).map(c => (c._1.replace(baseID, "").tail, c._2))

    val (theta_s, theta_t) = (base_kernel_state("spaceScale"), base_kernel_state("timeScale"))

    val op_state = config.filterNot(_._1.contains(baseID))

    val baseK = baseKernel.evaluateAt(base_kernel_state) _

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

object SE1dDiffusionKernel {

}