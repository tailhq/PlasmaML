package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.{analysis, utils}
import io.github.mandar2812.PlasmaML.utils._
import io.github.mandar2812.dynaml.analysis.DifferentiableMap
import io.github.mandar2812.dynaml.pipes._

/**
  * Specifies a Brautingham &amp; Albert type
  * trend function for radial diffusion priors.
  *
  * h(l, t) = (&alpha; l<sup>&beta;</sup> + &gamma;) 10<sup>b &times; Kp(t)</sup>
  *
  * @author mandar2812 date 27/06/2017.
  * */
case class MagnetosphericProcessTrend[T](Kp: DataPipe[Double, Double])(
  val transform: Encoder[T, (Double, Double, Double, Double)])
  extends MetaPipe[T, (Double, Double), Double] {

  self =>

  import MagnetosphericProcessTrend.MAGProcess

  override def run(data: T) = {
    val (alpha, beta, gamma, b) = transform(data)

    DataPipe((lt: (Double, Double)) => {
      val (l, t) = lt

      val kp = Kp(t)
      (math.exp(alpha)*math.pow(l, beta) + gamma)*math.pow(10d, b*kp)
    })

  }

  def gradL: MAGProcess[T] = MetaPipe(
    (p: T) => (lt: (Double, Double)) => {
      val (l, t) = lt
      val (alpha, beta, _, b) = transform(p)
      val kp = Kp(t)
      math.exp(alpha)*beta*math.pow(l, beta-1d)*math.pow(10d, b*kp)
    })

  def grad: Seq[MAGProcess[T]] = {

    val grad_alpha = MetaPipe(
      (p: T) => (lt: (Double, Double)) => {
        val (l, t) = lt
        val (alpha, beta, _, b) = transform(p)
        val kp = Kp(t)
        math.exp(alpha)*math.pow(l, beta)*math.pow(10d, b*kp)
      })

    val grad_beta = MetaPipe(
      (p: T) => (lt: (Double, Double)) => {
        val (l, t) = lt
        val (alpha, beta, _, b) = transform(p)
        val kp = Kp(t)
        math.exp(alpha)*math.log(l)*math.pow(l, beta)*math.pow(10d, b*kp)
      })

    val grad_gamma = MetaPipe(
      (p: T) => (lt: (Double, Double)) => {
        val (l, t) = lt
        val (_, _, _, b) = transform(p)
        val kp = Kp(t)
        math.pow(10d, b*kp)
      })

    val grad_b = MetaPipe(
      (p: T) => (lt: (Double, Double)) => {
        val (l, t) = lt
        val (alpha, beta, _, b) = transform(p)
        val kp = Kp(t)
        math.log(10.0)*kp*math.exp(alpha)*math.pow(l, beta)*math.pow(10d, b*kp)
      })

    Seq(grad_alpha, grad_beta, grad_gamma, grad_b)
  }

}

object MagnetosphericProcessTrend {

  type MAGProcess[T] = MetaPipe[T, (Double, Double), Double]

  type MAGParams = (Double, Double, Double, Double)

  def getEncoder(prefix: String): MagConfigEncoding = MagConfigEncoding(
    (prefix+"_alpha", prefix+"_beta", prefix+"_gamma", prefix+"_b")
  )

}

class MagTrend(override val Kp: DataPipe[Double, Double], val prefix: String) extends
  MagnetosphericProcessTrend[Map[String, Double]](Kp)(MagnetosphericProcessTrend.getEncoder(prefix)) {

  override val transform: MagConfigEncoding = MagnetosphericProcessTrend.getEncoder(prefix)

}

object MagTrend {
  def apply(Kp: DataPipe[Double, Double], prefix: String): MagTrend = new MagTrend(Kp, prefix)
}


class SimpleMagTrend(prefix: String) extends MagTrend(DataPipe((_: Double) => 0d), prefix)


abstract class MagParamBasis extends
  Basis[Double] with
  DifferentiableMap[Double, DenseVector[Double], DenseVector[Double]]


object MagParamBasis {

  def apply(func: Double => DenseVector[Double], grad_func: Double => DenseVector[Double]): MagParamBasis =
    new MagParamBasis{

      override protected val f: Double => DenseVector[Double] = func

      override def J(x: Double): DenseVector[Double] = grad_func(x)
    }

  val polynomial_basis: Int => MagParamBasis = n => apply(
    analysis.PolynomialBasisGenerator(n).run _,
    x => DenseVector.tabulate(n + 1)(i => i * math.pow(x, i - 1))
  )

  val chebyshev_basis: Int => MagParamBasis = n => apply(
    analysis.ChebyshevBasisGenerator(n, 1).run _,
    x => DenseVector(Array(0d) ++ Array.tabulate(n)(i => if(i+1 > 0) (i+1)*utils.chebyshev(i, x, kind = 2) else 0d))
  )

  val hermite_basis: Int => MagParamBasis = n => apply(
    analysis.HermiteBasisGenerator(n).run _,
    x => DenseVector(Array(0d) ++ (1 to n).toArray.map(i => i*utils.hermite(i - 1, x)))
  )

  val laguerre_basis: (Int, Double) => MagParamBasis = (n, alpha) => apply(
    (x: Double) => DenseVector((0 to n).toArray.map(i => laguerre(i, alpha, x))),
    (x: Double) => DenseVector((0 to n).toArray.map(i => grad_laguerre(1)(i, alpha, x)))
  )

}