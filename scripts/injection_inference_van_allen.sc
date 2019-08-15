import ammonite.ops._
import ammonite.ops.ImplicitWd._
import io.github.mandar2812.dynaml.repl.Router.main
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.tensorflow.dtfdata
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.probability.mcmc._
import io.github.mandar2812.dynaml.probability.GaussianRV
import io.github.mandar2812.PlasmaML.omni.{OMNIData => omni_data}
import io.github.mandar2812.PlasmaML.omni.{OMNILoader => omni_ops}
import io.github.mandar2812.PlasmaML.utils.DiracTuple2Kernel
import io.github.mandar2812.PlasmaML.dynamics.diffusion._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.MagParamBasis._
import io.github.mandar2812.PlasmaML.dynamics.diffusion.RDSettings._
import io.github.mandar2812.dynaml.pipes.{DataPipe, IterableDataPipe, Scaler}
import org.joda.time.{DateTime, DateTimeZone, Duration, Period}
import org.joda.time.format.DateTimeFormat

@main
def apply(
  data_path: Path = home / 'CWI / "data_psd_mageis.txt",
  start_time: DateTime = new DateTime(2013, 3, 1, 0, 0, 0),
  end_time: DateTime = new DateTime(2013, 4, 1, 0, 0, 0),
  basisSize: (Int, Int) = (5, 40),
  reg_data: Double = 2d,
  reg_galerkin: Double = 0.0001,
  quadrature_l: SGRadialDiffusionModel.QuadratureRule =
    SGRadialDiffusionModel.eightPointGaussLegendre,
  quadrature_t: SGRadialDiffusionModel.QuadratureRule =
    SGRadialDiffusionModel.eightPointGaussLegendre,
  burn: Int = 2000,
  num_post_samples: Int = 5000,
  num_bins_l: Int = 50,
  num_bins_t: Int = 100,
  basisCovFlag: Boolean = true,
  modelType: String = "hybrid"
): RDExperiment.Result[SGRadialDiffusionModel] = {

  println("Data Range ")
  println("------------")
  print("Start: ")
  pprint.pprintln(start_time)
  print("End: ")
  pprint.pprintln(end_time)

  val formatter = DateTimeFormat.forPattern("dd-MMM-yyyy HH:mm:ss")

  val process_date = DataPipe[String, DateTime](formatter.parseDateTime)

  DateTimeZone.setDefault(DateTimeZone.UTC)

  implicit val dateOrdering: Ordering[DateTime] = new Ordering[DateTime] {
    override def compare(x: DateTime, y: DateTime): Int =
      if (x.isBefore(y)) -1 else 1
  }

  val read_van_allen_data = fileToStream >
    trimLines >
    replaceWhiteSpaces >
    splitLine >
    IterableDataPipe[Array[String], (DateTime, (Double, Double))](
      (xs: Array[String]) => {
        val date = xs.take(2).mkString(" ")
        val data = xs.takeRight(2)
        (process_date(date), (data.head.toDouble, data.last.toDouble))
      }
    )

  val filter_van_allen_data = DataPipe[(DateTime, (Double, Double)), Boolean](
    p =>
      p._1.isAfter(start_time) && p._1
        .isBefore(end_time) && p._1.getMinuteOfHour == 0
  )

  val van_allen_data = dtfdata
    .dataset(Iterable(data_path.toString()))
    .flatMap(read_van_allen_data)
    .filter(filter_van_allen_data)

  println(s"Data set size = ${van_allen_data.size} patterns")

  val time_limits_van_allen =
    (van_allen_data.data.minBy(_._1), van_allen_data.data.maxBy(_._1))

  val read_kp_data =
    omni_ops.omniFileToStream(omni_data.Quantities.Kp, Seq()) >
      omni_ops.processWithDateTime >
      IterableDataPipe((xs: (DateTime, Seq[Double])) => (xs._1, xs._2.head))

  val filter_kp_data = DataPipe[(DateTime, Double), Boolean](
    p =>
      p._1.isAfter(time_limits_van_allen._1._1.minusHours(1)) && p._1.isBefore(
        time_limits_van_allen._2._1
      )
  )

  val process_time_stamp = DataPipe[DateTime, Int](d => {

    val period = new Duration(time_limits_van_allen._1._1, d)
    period.getStandardHours.toInt

  })

  val omni_files = {
    val start_year = start_time.getYear
    val end_year   = end_time.getYear

    (start_year to end_year).toIterable
      .map(y => pwd / 'data / omni_data.getFilePattern(y))
      .map(_.toString)
  }

  val kp_data = dtfdata
    .dataset(omni_files)
    .flatMap(read_kp_data)
    .filter(filter_kp_data)
    .map(process_time_stamp * Scaler((x: Double) => x / 10.0))

  val (tmin, tmax) = (kp_data.data.minBy(_._1)._1, kp_data.data.maxBy(_._1)._1)

  val kp_map = kp_data.data.toMap

  val scale_time   = Scaler((t: Double) => 5 * (t - tmin) / (tmax - tmin))
  val rescale_time = Scaler((t: Double) => t * (tmax - tmin) / 5 + tmin)

  val compute_kp = DataPipe[Double, Double]((t: Double) => {
    if (t <= tmin) kp_map.minBy(_._1)._2
    else if (t >= tmax) kp_map.maxBy(_._1)._2
    else {
      val (lower, upper) =
        (kp_map(math.floor(t).toInt), kp_map(math.ceil(t).toInt))

      if (math.ceil(t) == math.floor(t)) lower
      else
        lower + (t - math.floor(t)) * (upper - lower) / (math.ceil(t) - math
          .floor(t))
    }

  })

  val kp = compute_kp

  val psd_min = van_allen_data.data.minBy(_._2._2)._2._2

  val training_data = van_allen_data
    .map(
      process_time_stamp * (identityPipe[Double] * Scaler(
        (p: Double) => p / psd_min
      ))
    )
    .map(
      (p: (Int, (Double, Double))) => ((p._2._1, p._1.toDouble), p._2._2)
    )

  //println("Training data ")
  //pprint.pprintln(training_data.data)

  val timeLimits = (0d, tmax.toDouble)

  val lShellLimits = (1d, 7d)

  val nL = num_bins_l
  val nT = num_bins_t

  val rd_domain = RDDomain(
    Domain(lShellLimits),
    nL,
    Domain(timeLimits),
    nT
  )

  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_imq_basis(
    1d,
    lShellLimits,
    basisSize._1,
    timeLimits,
    basisSize._2,
    kind = 1
  )

  val seKernel =
    new GenExpSpaceTimeKernel[Double](1d, deltaL(rd_domain), deltaT(rd_domain))(
      sqNormDouble,
      l1NormDouble
    )

  val noiseKernel = new DiracTuple2Kernel(1d)

  noiseKernel.block_all_hyper_parameters

  val initial_config = (
    new Uniform(-10d, 10d).draw,
    new Uniform(0d, 5d).draw,
    0d,
    new Uniform(0d, 2d).draw
  )

  val model = if (modelType == "pure") {
    new GalerkinRDModel(
      kp,
      defaults.dll_params.values,
      defaults.lambda_params.values,
      initial_config
    )(
      seKernel,
      noiseKernel,
      training_data.data.toStream,
      chebyshev_hybrid_basis,
      lShellLimits,
      timeLimits,
      quadrature_l,
      quadrature_t,
      basisCovFlag = basisCovFlag
    )
  } else {
    new SGRadialDiffusionModel(
      kp,
      defaults.dll_params.values,
      defaults.lambda_params.values,
      initial_config
    )(
      seKernel,
      noiseKernel,
      training_data.data.toStream,
      chebyshev_hybrid_basis,
      lShellLimits,
      timeLimits,
      quadrature_l,
      quadrature_t,
      basisCovFlag = basisCovFlag
    )
  }

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(
        c =>
          c.contains("dll") ||
            c.contains("base::") ||
            c.contains("lambda_") ||
            c.contains("_gamma")
      )
  }

  model.block(blocked_hyp: _*)

  val hyp = model.effective_hyper_parameters

  val h_prior = RDExperiment.hyper_prior(hyp)

  model.regCol = reg_galerkin
  model.regObs = reg_data

  //Create the MCMC sampler
  val mcmc_sampler =
    new AdaptiveHyperParameterMCMC[SGRadialDiffusionModel, ContinuousDistr[
      Double
    ]](
      model,
      h_prior,
      burn
    )

  //Draw samples from the posterior
  val posterior_samples = mcmc_sampler.iid(num_post_samples).draw

  val resPath = RDExperiment.writeResults(
    rd_domain,
    kp,
    training_data.data.toStream,
    model.ghost_points,
    h_prior,
    posterior_samples,
    basisSize,
    s"ChebyshevIMQ[beta=1]",
    (model.regCol, model.regObs)
  )

  cp.into(data_path, resPath)

  RDExperiment.visualiseResultsInjection(
    if (num_post_samples > 5000) posterior_samples.takeRight(5000)
    else posterior_samples,
    Map(),
    h_prior
  )

  RDExperiment.samplingReport(
    posterior_samples.map(_.filterKeys(quantities_injection.contains)),
    hyp
      .filter(quantities_injection.contains)
      .map(c => (c, quantities_injection(c)))
      .toMap,
    Map(),
    mcmc_sampler.sampleAcceptenceRate,
    "injection"
  )

  val scriptPath = pwd / "mag-core" / 'scripts / "visualiseResultsVanAllen.R"

  try {
    %%('Rscript, scriptPath.toString, resPath.toString, "injection")
  } catch {
    case e: ammonite.ops.ShelloutException => pprint.pprintln(e)
  }

  RDExperiment.Result(
    van_allen_data,
    training_data,
    kp,
    model,
    h_prior,
    mcmc_sampler,
    posterior_samples,
    resPath
  )

}
