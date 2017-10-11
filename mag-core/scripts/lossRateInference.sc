{
  import breeze.linalg._
  import breeze.numerics.Bessel
  import breeze.stats.distributions._
  import spire.implicits._
  import com.quantifind.charts.Highcharts._

  import io.github.mandar2812.dynaml.utils._
  import io.github.mandar2812.dynaml.kernels._
  import io.github.mandar2812.dynaml.pipes.DataPipe
  import io.github.mandar2812.dynaml.probability._
  import io.github.mandar2812.dynaml.probability.distributions._
  import io.github.mandar2812.dynaml.probability.mcmc._

  import io.github.mandar2812.PlasmaML.dynamics.diffusion._
  import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel

  import io.github.mandar2812.PlasmaML.dynamics.diffusion.GPRadialDiffusionModel



  val (nL,nT) = (200, 50)

  val lShellLimits = (1.0, 7.0)
  val timeLimits = (0.0, 5.0)

  val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

  val (lShellVec, timeVec) = rds.stencil

  val Kp = DataPipe((t: Double) => {
    if(t<= 0d) 2.5
    else if(t < 1.5) 2.5 + 4*t
    else if (t >= 1.5 && t< 3d) 8.5
    else if(t >= 3d && t<= 5d) 17.5 - 3*t
    else 2.5
  })

  val rowSelectorRV = MultinomialRV(DenseVector.fill[Double](lShellVec.length)(1d/lShellVec.length.toDouble))
  val colSelectorRV = MultinomialRV(DenseVector.fill[Double](timeVec.length-1)(1d/timeVec.length.toDouble))

  val measurement_noise = GaussianRV(0.0, 0.5)
  /*
   * Define parameters of radial diffusion system:
   *
   *  1) The diffusion field: dll
   *  2) The particle injection process: q
   *  3) The loss parameter: lambda
   *
   * Using the MagnetosphericProcessTrend class,
   * we define each unknown process using a canonical
   * parameterization of diffusion processes in the
   * magnetosphere.
   *
   * For each process we must specify 4 parameters
   * alpha, beta, a, b
   * */

  //Diffusion Field
  val dll_alpha = 0d//1d
  val dll_beta = 10d
  val dll_gamma = 0d
  val dll_a = -9.325
  val dll_b = 0.506

  //Loss Process
  val lambda_alpha = math.log(math.pow(10d, -4)/2.4)
  val lambda_beta = 1.0d
  val lambda_a = 2.5d
  val lambda_b = 0.18

  val (q_alpha, q_beta, q_gamma, q_b) = (Double.NegativeInfinity, 0d, 0d, 0d)

  //Create ground truth diffusion parameter functions
  val dll = (l: Double, t: Double) =>
    math.exp(dll_alpha)*math.pow(l, dll_beta)*math.pow(10, dll_a + dll_b*Kp(t))

  val Q = (l: Double, t: Double) => 0d//(math.exp(q_alpha)*math.pow(l, q_beta) + q_gamma)*math.pow(10, q_b*Kp(t))

  val lambda = (l: Double, t: Double) =>
    math.exp(lambda_alpha)*math.pow(l, lambda_beta)*math.pow(10, lambda_a + lambda_b*Kp(t))

  val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)

  val initialPSD = (l: Double) => Bessel.i1(omega*(l - lShellLimits._1))*1E2

  //Create ground truth PSD data and corrupt it with statistical noise.
  val groundTruth = rds.solve(Q, dll, lambda)(initialPSD)
  val ground_truth_matrix = DenseMatrix.horzcat(groundTruth.map(_.asDenseMatrix.t):_*)


  val noise_mat = DenseMatrix.tabulate[Double](nL+1, nT+1)((_, _) => measurement_noise.draw)
  val data: DenseMatrix[Double] = ground_truth_matrix + noise_mat

  val num_boundary_data = 10
  val num_bulk_data = 50
  val num_dummy_data = 100

  val coordinateIndices = combine(Seq(lShellVec.indices, timeVec.tail.indices)).map(s => (s.head, s.last))


  val boundary_data = lShellVec.indices
    .filter(_ => Rand.uniform.draw() <= num_boundary_data.toDouble/lShellVec.length)
    .map(lIndex => {
      ((lShellVec(lIndex),0d), data(lIndex, 0))
    }).toStream

  val bulk_data = coordinateIndices
    .filter(_ => Rand.uniform.draw() <= num_bulk_data.toDouble/coordinateIndices.length)
    .map((lt) => {
      val (l, t) = (lShellVec(lt._1), timeVec(lt._2+1))
      ((l,t), data(lt._1, lt._2+1))
    }).toStream

  val gp_data: Stream[((Double, Double), Double)] = boundary_data ++ bulk_data

  val psdVar = getStats(gp_data.map(p => DenseVector(p._1._1)).toList)._2(0)
  println("PSD Observational Variance = "+psdVar)


  val imq_basis = new InverseMQPSDBasis(1d)(
    lShellLimits, 20, timeLimits, 20, (false, false)
  )

  val mq_basis = new MQPSDBasis(
    lShellLimits, 20, timeLimits, 20, (false, false)
  )

  val colocation_points: Stream[(Double, Double)] = {
    (0 until num_dummy_data).map(_ => {
      val rowS = rowSelectorRV.draw
      val colS = colSelectorRV.draw
      //val (l, t) = (lShellVec(rowS), timeVec(colS))
      //((l,t), measurement_noise.draw)
      (lShellVec(rowS), timeVec(colS))
    }).toStream

    //imq_basis._centers.toStream
  }

  val burn = 4000
  //Create the GP PDE model

  val gpKernel = new GenExpSpaceTimeKernel[Double](
    psdVar, rds.deltaL, rds.deltaT)(
    sqNormDouble, l1NormDouble)

  val noiseKernel = new DiracTuple2Kernel(0.5)

  noiseKernel.block_all_hyper_parameters

  val model = new GPRadialDiffusionModel(
    Kp,
    (math.log(math.exp(dll_alpha)*math.pow(10d, dll_a)), dll_beta, dll_gamma, dll_b),
    (0d, 0.2, 0d, 0.0),
    (q_alpha, q_beta, q_gamma, q_b))(
    gpKernel, noiseKernel,
    gp_data, colocation_points,
    mq_basis
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(
        c => c.contains("dll") || c.contains("base::") || c.contains("tau_gamma") || c.contains("Q_")
      )
  }


  model.block(blocked_hyp:_*)
  //Create the MCMC sampler
  val hyp = model.effective_hyper_parameters

  val hyper_prior = {
    hyp.filter(_.contains("base::")).map(h => (h, new LogNormal(0d, 2d))).toMap ++
    hyp.filterNot(h => h.contains("base::") || h.contains("tau")).map(h => (h, new Gaussian(0d, 2.5d))).toMap ++
    Map(
      "tau_alpha" -> new Gaussian(0d, 1d),
      "tau_beta" -> TruncatedGaussian(1d, 1d, 0d, 3.5d),
      "tau_b" -> new Gaussian(0d, 2.0))
  }

  model.regCol = 0.001d
  model.regObs = 0.1d

  val mcmc_sampler = new AdaptiveHyperParameterMCMC[
    model.type, ContinuousDistr[Double]](
    model, hyper_prior, burn)

  val num_post_samples = 4000

  //Draw samples from the posterior
  val samples = mcmc_sampler.iid(num_post_samples).draw

  val post_vecs = samples.map(c => DenseVector(c("tau_alpha"), c("tau_beta"), c("tau_b")))
  val post_moments = getStats(post_vecs.toList)

  val quantities = Map("tau_alpha" -> 0x03B1.toChar, "tau_beta" -> 0x03B2.toChar, "tau_b" -> 'b')

  val gt = Map(
    "tau_alpha" -> math.log(math.exp(lambda_alpha)*math.pow(10d, lambda_a)),
    "tau_beta" -> lambda_beta,
    "tau_b" -> lambda_b,
    "Q_alpha" -> math.log(math.exp(q_alpha)),
    "Q_beta" -> q_beta,
    "Q_gamma" -> q_gamma,
    "Q_b" -> q_b)

  println("\n:::::: MCMC Sampling Report ::::::\n")

  println("Quantity: "+0x03C4.toChar+"(l,t) = "+0x03B1.toChar+"l^("+0x03B2.toChar+")*10^(b*K(t))")

  println("Markov Chain Acceptance Rate = "+mcmc_sampler.sampleAcceptenceRate)

  quantities.zipWithIndex.foreach(c => {
    val ((key, char), index) = c
    println("\n------------------------------")
    println("Parameter: "+char)
    println("Ground Truth:- "+gt(key))
    println("Posterior Moments: mean = "+post_moments._1(index)+" variance = "+post_moments._2(index))
  })


  val lMax = 10
  val tMax = 10

  val solution = groundTruth

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
    if(i < nL) lShellLimits._1+(rds.deltaL*i*20)
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
    if(i < nL) timeLimits._1+(rds.deltaT*i*5)
    else timeLimits._2).toArray
    .toSeq.map(s => "t = "+"%3f".format(s)) /*++ Seq.tabulate(tMax)(t => "Mean t="+"%3f".format(timeVec(5*t)))*/)

  title("Variation of Phase Space Density f(L,t)")
  xAxis("L")
  yAxis("f(L,t)")


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

  histogram(samples.map(_("tau_beta")), 100)
  hold()
  histogram((1 to num_post_samples).map(_ => hyper_prior("tau_beta").draw), 100)
  legend(Seq("Posterior Samples", "Prior Samples"))
  unhold()
  title("Histogram: "+0x03B2.toChar)

}
