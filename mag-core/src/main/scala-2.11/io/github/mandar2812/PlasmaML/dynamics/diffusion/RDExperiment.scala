package io.github.mandar2812.PlasmaML.dynamics.diffusion

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.Bessel
import breeze.stats.distributions._
import com.quantifind.charts.Highcharts.{histogram, hold, legend, line, scatter, spline, title, unhold, xAxis, yAxis}
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.{GaussianRV, MultinomialRV, RandomVariable}
import io.github.mandar2812.dynaml.utils.{combine, getStats}

/**
  * <h3>Radial Diffusion Experiments</h3>
  *
  * A set of convenient work-flows and functions for setting
  * up inference routines in the radial diffusion setting.
  * */
object RDExperiment {

  /**
    * Create the radial diffusion solver
    * from the domain descriptions.
    *
    * @param lShellLimits The lower and upper limits of L-shell
    * @param timeLimits The lower and upper limits of time
    * @param nL The number bins in L-shell
    * @param nT Number of bins in time
    * */
  def solver(
    lShellLimits: (Double, Double),
    timeLimits: (Double, Double),
    nL: Int, nT: Int): RadialDiffusion =
    new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

  def generateData(
    rds: RadialDiffusion,
    dll: (Double, Double) => Double,
    lambda: (Double, Double) => Double,
    q: (Double, Double) => Double,
    initialPSD: (Double) => Double)(
    noise: RandomVariable[Double],
    num_boundary_points: Int,
    num_bulk_points: Int,
    num_colocation_points: Int) = {

    val groundTruth = rds.solve(q, dll, lambda)(initialPSD)
    val ground_truth_matrix = DenseMatrix.horzcat(groundTruth.map(_.asDenseMatrix.t):_*)


    val noise_mat = DenseMatrix.tabulate[Double](rds.nL+1, rds.nT+1)((_, _) => noise.draw)
    val data: DenseMatrix[Double] = ground_truth_matrix + noise_mat

    val (lShellVec, timeVec) = rds.stencil

    val coordinateIndices = combine(Seq(lShellVec.indices, timeVec.tail.indices)).map(s => (s.head, s.last))


    val boundary_data = lShellVec.indices
      .filter(_ => Rand.uniform.draw() <= num_boundary_points.toDouble/lShellVec.length)
      .map(lIndex => {
        ((lShellVec(lIndex),0d), data(lIndex, 0))
      }).toStream

    val bulk_data = coordinateIndices
      .filter(_ => Rand.uniform.draw() <= num_bulk_points.toDouble/coordinateIndices.length)
      .map((lt) => {
        val (l, t) = (lShellVec(lt._1), timeVec(lt._2+1))
        ((l,t), data(lt._1, lt._2+1))
      }).toStream

    val gp_data: Stream[((Double, Double), Double)] = boundary_data ++ bulk_data

    val rowSelectorRV = MultinomialRV(DenseVector.fill[Double](lShellVec.length)(1d/lShellVec.length.toDouble))
    val colSelectorRV = MultinomialRV(DenseVector.fill[Double](timeVec.length-1)(1d/timeVec.length.toDouble))

    val colocation_points: Stream[(Double, Double)] = {
      (0 until num_colocation_points).map(_ => {
        val rowS = rowSelectorRV.draw
        val colS = colSelectorRV.draw
        (lShellVec(rowS), timeVec(colS))
      }).toStream
    }

    (groundTruth, gp_data, colocation_points)
  }

  def hyper_prior(hyp: List[String]) = {
    hyp.filter(_.contains("base::")).map(h => (h, new LogNormal(0d, 2d))).toMap ++
      hyp.filterNot(h => h.contains("base::") || h.contains("tau")).map(h => (h, new Gaussian(0d, 2.5d))).toMap ++
      Map(
        "tau_alpha" -> new Gaussian(0d, 1d),
        "tau_beta" -> new Gamma(2d, 2d),
        "tau_b" -> new Gaussian(0d, 2.0)).filterKeys(hyp.contains _)
  }

  def samplingReport(
    samples: Stream[Map[String, Double]],
    quantities: Map[String, Char],
    gt: Map[String, Double],
    acceptanceRate: Double): Unit = {

    val post_vecs = samples.map(c => DenseVector(c.values.toArray))

    val post_moments = getStats(post_vecs.toList)


    println("\n:::::: MCMC Sampling Report ::::::\n")

    println("Quantity: "+0x03C4.toChar+"(l,t) = "+0x03B1.toChar+"l^("+0x03B2.toChar+")*10^(b*K(t))")

    println("Markov Chain Acceptance Rate = "+acceptanceRate)

    quantities.zipWithIndex.foreach(c => {
      val ((key, char), index) = c
      println("\n------------------------------")
      println("Parameter: "+char)
      println("Ground Truth:- "+gt(key))
      println("Posterior Moments: mean = "+post_moments._1(index)+" variance = "+post_moments._2(index))
    })
  }

  def visualisePSD(
    lShellLimits: (Double, Double),
    timeLimits: (Double, Double), nL: Int, nT: Int)(
    initialPSD: (Double) => Double,
    solution: Stream[DenseVector[Double]],
    Kp: DataPipe[Double, Double]): Unit = {

    val lMax = 10
    val tMax = 10

    val (lShellVec, timeVec) = RadialDiffusion.buildStencil(lShellLimits, nL, timeLimits, nT)


    /*
     *
     * First set of plots
     *
     * */

    spline(timeVec.zip(solution.map(_(0))))
    hold()

    (0 until lMax).foreach(l => {
      spline(timeVec.zip(solution.map(_(l*20))))
    })

    unhold()

    legend(DenseVector.tabulate[Double](lMax+1)(i =>
      if(i < nL) lShellLimits._1+(deltaL*i*20)
      else lShellLimits._2).toArray.map(s => "L = "+"%3f".format(s)))
    title("Evolution of Phase Space Density f(L,t)")
    xAxis("time")
    yAxis("f(L,t)")



    line(timeVec.map(t => (t, Kp(t))).toSeq)
    xAxis("time")
    yAxis("Kp")
    title("Evolution of Kp")

    spline(lShellVec.map(l => (l, initialPSD(l))).toSeq)
    xAxis("L")
    yAxis("f(L, 0)")
    title("Phase Space Density Profile, t = 0")


    /*
     *
     * Second set of plots
     *
     * */
    spline(lShellVec.toArray.toSeq.zip(solution.head.toArray.toSeq))
    hold()

    (0 until tMax).foreach(t => {
      spline(lShellVec.toArray.toSeq.zip(solution(t*5).toArray.toSeq))
    })
    unhold()

    legend(DenseVector.tabulate[Double](tMax+1)(i =>
      if(i < nL) timeLimits._1+(deltaT*i*5)
      else timeLimits._2).toArray
      .toSeq.map(s => "t = "+"%3f".format(s)))

    title("Variation of Phase Space Density f(L,t)")
    xAxis("L")
    yAxis("f(L,t)")

  }

  def visualiseResultsLoss(
    samples: Stream[Map[String, Double]],
    gt: Map[String, Double],
    hyper_prior: Map[String, ContinuousDistr[Double]]): Unit = {

    scatter(samples.map(c => (c("tau_alpha"), c("tau_b"))))
    hold()
    scatter(Seq((gt("tau_alpha"), gt("tau_b"))))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples:- "+0x03B1.toChar+" vs b")
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": b")
    unhold()

    scatter(samples.map(c => (c("tau_alpha"), c("tau_beta"))))
    hold()
    scatter(Seq((gt("tau_alpha"), gt("tau_beta"))))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples "+0x03B1.toChar+" vs "+0x03B2.toChar)
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": "+0x03B2.toChar)
    unhold()

    histogram(samples.map(_("tau_beta")), 50)
    hold()
    histogram((1 to samples.length).map(_ => hyper_prior("tau_beta").draw), 100)
    legend(Seq("Posterior Samples", "Prior Samples"))
    unhold()
    title("Histogram: "+0x03B2.toChar)
  }

  def visualiseResultsInjection(
    samples: Stream[Map[String, Double]],
    gt: Map[String, Double],
    hyper_prior: Map[String, ContinuousDistr[Double]]): Unit = {


    scatter(samples.map(c => (c("Q_alpha"), c("Q_b"))))
    hold()
    scatter(Seq((gt("Q_alpha"), gt("Q_b"))))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples:- "+0x03B1.toChar+" vs b")
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": b")
    unhold()



    scatter(samples.map(c => (c("Q_alpha"), c("Q_beta"))))
    hold()
    scatter(Seq((gt("Q_alpha"), gt("Q_beta"))))
    legend(Seq("Posterior Samples", "Ground Truth"))
    title("Posterior Samples "+0x03B1.toChar+" vs "+0x03B2.toChar)
    xAxis(0x03C4.toChar+": "+0x03B1.toChar)
    yAxis(0x03C4.toChar+": "+0x03B2.toChar)
    unhold()

    histogram(samples.map(_("Q_b")), 100)
    hold()
    histogram((1 to samples.length).map(_ => hyper_prior("Q_b").draw), 100)
    legend(Seq("Posterior Samples", "Prior Samples"))
    unhold()
    title("Histogram: "+"b")

  }



}

object RDSettings {

  /*Define the data generation primitives*/
  var (nL,nT) = (200, 50)

  var lShellLimits: (Double, Double) = (1.0, 7.0)
  var timeLimits: (Double, Double) = (0.0, 5.0)

  def deltaL = (lShellLimits._2 - lShellLimits._1)/nL

  def deltaT = (timeLimits._2 - timeLimits._1)/nT

  var dll_params: (Double, Double, Double, Double) = (
    math.log(math.exp(0d)*math.pow(10d, -9.325)),
    10d, 0d, 0.506)

  var lambda_params: (Double, Double, Double, Double) = (
    math.log(math.pow(10d, -4)*math.pow(10d, 2.5d)/2.4),
    1d, 0d, 0.18)

  var q_params: (Double, Double, Double, Double) = (Double.NegativeInfinity, 0d, 0d, 0d)

  val quantities_loss = Map(
    "tau_alpha" -> 0x03B1.toChar,
    "tau_beta" -> 0x03B2.toChar,
    "tau_b" -> 'b')

  val quantities_injection = Map(
    "Q_alpha" -> 0x03B1.toChar,
    "Q_beta" -> 0x03B2.toChar,
    "Q_gamma" -> 0x03B3.toChar,
    "Q_b" -> 'b'
  )

  def gt = Map(
    "tau_alpha" -> lambda_params._1,
    "tau_beta" -> lambda_params._2,
    "tau_b" -> lambda_params._4,
    "Q_alpha" -> q_params._1,
    "Q_beta" -> q_params._2,
    "Q_gamma" -> q_params._3,
    "Q_b" -> q_params._4)

  var Kp = DataPipe((t: Double) => {
    if(t<= 0d) 2.5
    else if(t < 1.5) 2.5 + 4*t
    else if (t >= 1.5 && t< 3d) 8.5
    else if(t >= 3d && t<= 5d) 17.5 - 3*t
    else 2.5
  })

  def dll = (l: Double, t: Double) =>
    math.exp(dll_params._1)*math.pow(l, dll_params._2)*math.pow(10, dll_params._4*Kp(t))

  def Q = (l: Double, t: Double) =>
    (math.exp(q_params._1)*math.pow(l, q_params._2) + q_params._3)*math.pow(10, q_params._4*Kp(t))

  def lambda = (l: Double, t: Double) =>
    math.exp(lambda_params._1)*math.pow(l, lambda_params._2)*math.pow(10, lambda_params._4*Kp(t))

  protected def omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)

  var initialPSD = (l: Double) => Bessel.i1(omega*(l - lShellLimits._1))*1E2

  var measurement_noise = GaussianRV(0.0, 0.5)


  var num_boundary_data = 50
  var num_bulk_data = 100
  var num_dummy_data = 200

  var regData = 0.01

  var regColocation = 1E-8

  var lMax = 10
  var tMax = 10


}
