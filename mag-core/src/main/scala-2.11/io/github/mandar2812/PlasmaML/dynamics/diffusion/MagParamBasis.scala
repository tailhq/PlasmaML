package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.PlasmaML.utils.{grad_laguerre, laguerre}
import io.github.mandar2812.dynaml.analysis.DifferentiableMap
import io.github.mandar2812.dynaml.pipes.Basis
import io.github.mandar2812.dynaml.{analysis, utils}

abstract class MagParamBasis extends
  Basis[Double] with
  DifferentiableMap[Double, DenseVector[Double], DenseVector[Double]]


object MagParamBasis {

  def apply(func: Double => DenseVector[Double], grad_func: Double => DenseVector[Double]): MagParamBasis =
    new MagParamBasis {

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