{
  import breeze.linalg._
  import breeze.stats.distributions._
  import com.quantifind.charts.Highcharts._
  import io.github.mandar2812.PlasmaML.dynamics.diffusion._
  import io.github.mandar2812.dynaml.utils._
  import io.github.mandar2812.dynaml.DynaMLPipe._
  import io.github.mandar2812.dynaml.kernels.MAKernel
  import io.github.mandar2812.dynaml.models.gp.GPOperatorModel
  import io.github.mandar2812.dynaml.pipes.DataPipe
  import io.github.mandar2812.dynaml.probability.mcmc._
  import io.github.mandar2812.dynaml.probability._
  import io.github.mandar2812.dynaml.analysis.VectorField
  import scala.util.Random

  val (nL,nT) = (200, 50)

  val lMax = 7
  val tMax = 5

  val lShellLimits = (1.0, 7.0)
  val timeLimits = (0.0, 5.0)

  val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)
  val theta = 0.06
  val gamma = 0.001 + theta*math.pow(omega*lShellLimits._2, 2.0)

  val fl = (l: Double, _: Double) => math.sin(omega*(l - lShellLimits._1))
  val ft = (_: Double, t: Double) => math.exp(-gamma*t)

  val referenceSolution = (l: Double, t: Double) => fl(l,t)*ft(l,t)

  val rds = new RadialDiffusion(lShellLimits, timeLimits, nL, nT)

  val (lShellVec, timeVec) = rds.stencil

  val q = (l: Double, t: Double) => 0.0
  val dll = (l: Double, _: Double) => theta*l*l
  val loss = (l: Double, _: Double) => gamma - math.pow(l*omega, 2.0)*theta

  val initialPSD = (l: Double) => referenceSolution(l, 0.0)

  val Kp = DataPipe((_: Double) => 0d)

  val measurement_noise = GaussianRV(0.0, 0.01)
  val lShellRV = new Uniform(lShellLimits._1, lShellLimits._2)
  val tRV = new Uniform(timeLimits._1, timeLimits._2)

  val num_data = 5

  val psd_data: Seq[((Double, Double), Double)] = {
    (0 until num_data).map(_ => {

      val (l, t) = (lShellRV.draw(), tRV.draw())
      ((l,t), referenceSolution(l, t) + measurement_noise.draw)
    }) ++ (0 until num_data).map(_ => {
      val (l, t) = (if(Random.nextDouble() <= 0.5) lShellLimits._2 else lShellLimits._1, tRV.draw())
      ((l,t), referenceSolution(l, t) + measurement_noise.draw)
    })
  }

  val (data_features, psd_targets) = (psd_data.map(_._1), DenseVector(psd_data.toArray.map(_._2)))

}
{
  val burn = 5000
  val gpKernel =
    new SE1dExtRadDiffusionKernel(
      1.0, 0.15, 0.05, Kp)(
      (theta, 2d, 0d, 0d),
        (0.5, 2d, 1d, 0d),
        "L2", "L2"
    )

  val noiseKernel = new MAKernel(math.sqrt(0.01))

  noiseKernel.block_all_hyper_parameters

  val blocked_hyp = {
    gpKernel.hyper_parameters.filter(h => h.contains("dll") || h.contains("base::")) ++
    Seq("tau_b")
  }

  gpKernel.block(blocked_hyp:_*)

  implicit val dataT = identityPipe[Seq[((Double, Double), Double)]]

  val psdMean = psd_data.map(_._2).sum/psd_data.length

  val model = GPOperatorModel[Seq[((Double, Double), Double)], Double, SE1dExtRadDiffusionKernel](
    gpKernel, noiseKernel:*noiseKernel, DataPipe((lt: (Double, Double)) => psdMean))(
    psd_data, psd_data.length)


  //Create the MCMC sampler
  val hyp = gpKernel.effective_hyper_parameters ++ noiseKernel.effective_hyper_parameters

  val hyper_prior = {
    hyp.filter(_.contains("base::")).map(h => (h, new LogNormal(0d, 2d))).toMap ++
    hyp.filterNot(h => h.contains("base::") || h.contains("tau")).map(h => (h, new Gaussian(0d, 2.5d))).toMap ++
    Map(
      "tau_alpha" -> new Gaussian(0d, 2.5d),
      "tau_beta" -> new LogNormal(0d, 2d),
      "tau_gamma" -> new Gaussian(0d, 2.5))
  }

  val num_hyp = hyp.length
  implicit val ev = VectorField(num_hyp)


  val mcmc = new HyperParameterSCAM[model.type, ContinuousDistr[Double]](
    model, hyper_prior, burn)

  //Draw samples from the posteior

  val samples: Stream[Map[String, Double]] = mcmc.iid(2000).draw

  val post_beta_tau = samples.map(c => c("tau_beta")).sum/2000d

  val post_vecs = samples.map(c => DenseVector(c("tau_alpha"), c("tau_beta"), c("tau_gamma")))
  val post_moments = getStats(post_vecs.toList)

  val quantities = Map("alpha" -> 0x03B1.toChar, "beta" -> 0x03B2.toChar, "gamma" -> 0x03B3.toChar)
  val gt = Map("alpha" -> -omega*omega*theta, "beta" -> 2d, "gamma" -> gamma)


}
{
  println("\n:::::: MCMC Sampling Report ::::::")

  println("Quantity: "+0x03C4.toChar+"(l,t) = "+0x03B1.toChar+"l^("+0x03B2.toChar+")*10^(a + b*K(t))")


  quantities.zipWithIndex.foreach(c => {
    val ((key, char), index) = c
    println("\n------------------------------")
    println("Parameter: "+char)
    println("Ground Truth:- "+gt(key))
    println("Posterior Moments: mean = "+post_moments._1(index)+" variance = "+post_moments._2(index))
  })

}

{
  scatter(samples.map(c => (c("tau_alpha"), c("tau_beta"))))
  hold()
  scatter(Seq((-omega*omega*theta, 2.0)))
  unhold()
  legend(Seq("Posterior Samples", "Ground Truth"))
  xAxis(0x03C4.toChar+": "+0x03B1.toChar)
  yAxis(0x03C4.toChar+": "+0x03B2.toChar)
  title("Posterior Samples "+0x03B1.toChar+" vs "+0x03B2.toChar)

  scatter(samples.map(c => (c("tau_gamma"), c("tau_beta"))))
  hold()
  scatter(Seq((gamma, 2.0)))
  unhold()
  legend(Seq("Posterior Samples", "Ground Truth"))
  xAxis(0x03C4.toChar+": "+0x03B3.toChar)
  yAxis(0x03C4.toChar+": "+0x03B2.toChar)
  title("Posterior Samples "+0x03B3.toChar+" vs "+0x03B2.toChar)

  scatter(samples.map(c => (c("tau_alpha"), c("tau_gamma"))))
  hold()
  scatter(Seq((-omega*omega*theta, gamma)))
  unhold()
  legend(Seq("Posterior Samples", "Ground Truth"))
  xAxis(0x03C4.toChar+": "+0x03B1.toChar)
  yAxis(0x03C4.toChar+": "+0x03B3.toChar)
  title("Posterior Samples "+0x03B1.toChar+" vs "+0x03B3.toChar)

}

