package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.neuralnets.{NeuralLayer, VectorLinear}
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}

/**
  *
  * Represents a Radial Diffusion forward propagation computation.
  *
  * f(n+1) = &Beta;(n)<sup>-1</sup>.(&Alpha;(n).f(n) + &gamma;(n))
  *
  * @author mandar2812 date 13/04/2017.
  *
  *
  * @param alpha The tri-diagonal portion (non-zero) of &Alpha;
  * @param beta The tri-diagonal portion (non-zero) of &Beta;
  * @param gamma The intercept &gamma;
  *
  * */
class RadialDiffusionLayer(
  alpha: Seq[Seq[Double]],
  beta: Seq[Seq[Double]],
  gamma: DenseVector[Double]) extends NeuralLayer[
  (Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double]),
  DenseVector[Double], DenseVector[Double]] {

  override val parameters = (alpha, beta, gamma)

  override val localField = DataPipe((x: DenseVector[Double]) => {

    //First calculate alpha*x + gamma
    val xArr = Array(0.0) ++ x.toArray ++ Array(0.0)

    val a = DenseVector(alpha.zip(xArr.toSeq.sliding(3).toSeq).map(
      couple => couple._1.zip(couple._2).map(c => c._1*c._2).sum
    ).toArray) + gamma

    //Return beta\(alpha*x+gamma)
    DenseVector(RadialDiffusionLayer.triDiagSolve(beta, a.toArray))
  })

  override val activationFunc = VectorLinear

  override val forward = localField > activationFunc
}


object RadialDiffusionLayer {

  /**
    * Solve a tri-diagonal linear system using
    * the <a href="https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm">Thomas algorithm</a>.
    *
    * &Alpha;.x = b
    *
    * @param a The tri-diagonal portion of matrix &Alpha;
    * @param b Right Hand Side of the linear system.
    *
    * */
  def triDiagSolve(a: Seq[Seq[Double]], b: Seq[Double]): Array[Double] = {


    /*
    * Does a forward pass of the Thomas algorithm in a tail recursive manner
    * returns a list of (c', d') in the reverse order.
    * */
    def forwardTRec(
      list: List[(Seq[Double], Double)],
      cAcc: Double, dAcc: Double,
      cpdpAcc: List[(Double, Double)]): List[(Double, Double)] = list match {

      case Nil => cpdpAcc
      case l::tail =>
        val (Seq(a, b, c), d) = l
        val (cp, dp) = (c/(b - a*cAcc), (d - a*dAcc)/(b - a*cAcc))
        forwardTRec(tail, cp, dp, (cp, dp) :: cpdpAcc)

    }

    /*
    * Does a reverse pass of the Thomas algorithm in a tail recursive manner
    * expects cpdp to be in reverse order see forwardTRec
    * */
    def reverseTRec(
      cpdp: List[(Double, Double)],
      xP: Double, xAcc: List[Double]): Seq[Double] = cpdp match {
      case Nil => xAcc
      case h::tail =>
        val (c, d) = h
        val xC = d - c*xP
        reverseTRec(tail, xC, xC :: xAcc)

    }

    val cdprime = forwardTRec(a.zip(b).toList, 0.0, 0.0, List())
    val x = reverseTRec(cdprime, 0.0, List())

    x.toArray
  }

  val forwardPropagate = MetaPipe(
    (params: (Seq[Seq[Double]], Seq[Seq[Double]], DenseVector[Double])) => (x: DenseVector[Double]) => {

    val (alpha, beta, gamma) = params

    //First calculate alpha*x + gamma
    val xArr = Array(0.0) ++ x.toArray ++ Array(0.0)

    val a = DenseVector(alpha.zip(xArr.toSeq.sliding(3).toSeq).map(
      couple => couple._1.zip(couple._2).map(c => c._1*c._2).sum
    ).toArray) + gamma

    DenseVector(RadialDiffusionLayer.triDiagSolve(beta, a.toArray))
  })

  /**
    * Convenience Method.
    * Create a Radial Diffusion forward propagation layer.
    *
    * f(n+1) = &Beta;(n)<sup>-1</sup>.(&Alpha;(n).f(n) + &gamma;(n))
    * @param alpha The tri-diagonal portion (non-zero) of &Alpha;
    * @param beta The tri-diagonal portion (non-zero) of &Beta;
    * @param gamma The intercept &gamma;
    *
    * @return A [[RadialDiffusionLayer]] instance representing the
    *         specified forward propagation computation.
    * */
  def apply(
    alpha: Seq[Seq[Double]],
    beta: Seq[Seq[Double]],
    gamma: DenseVector[Double]): RadialDiffusionLayer =
    new RadialDiffusionLayer(alpha, beta, gamma)
}
