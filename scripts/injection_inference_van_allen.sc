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
import io.github.mandar2812.dynaml.pipes.{DataPipe, IterableDataPipe, Scaler, StreamDataPipe}
import org.joda.time.{DateTime, DateTimeZone, Duration, Period}
import org.joda.time.format.DateTimeFormat


def apply(
  data_path: Path                             = home/'Downloads/"psd_data_tLf.txt",
  basisSize: (Int, Int)                       = (4, 4),
  reg_data: Double                            = 0.5,
  reg_galerkin: Double                        = 1.0,
  burn: Int                                   = 2000,
  num_post_samples: Int                       = 5000) = {

  val formatter = DateTimeFormat.forPattern("dd-MMM-yyyy HH:mm:ss")

  val process_date = DataPipe[String, DateTime](formatter.parseDateTime)

  DateTimeZone.setDefault(DateTimeZone.UTC)

  implicit val dateOrdering: Ordering[DateTime] = new Ordering[DateTime] {
    override def compare(x: DateTime, y: DateTime): Int = if(x.isBefore(y)) -1 else 1
  }

  val read_van_allen_data = fileToStream >
    dropHead >
    splitLine >
    IterableDataPipe[Array[String], (DateTime, (Double, Double))]((xs: Array[String]) => {
      (process_date(xs.head), (xs.tail.head.toDouble, xs.tail.last.toDouble))
    })

  val filter_van_allen_data = DataPipe[(DateTime, (Double, Double)), Boolean](p => p._1.getMinuteOfHour == 0)

  //val group_van_allen_by_hour = DataPipe((p: Iterable[(DateTime, (Double, Double))]) => p.groupBy(_._1))

  val van_allen_data = dtfdata.dataset(Iterable(data_path.toString()))
    .flatMap(read_van_allen_data)
    .filter(filter_van_allen_data)

  val time_limits_van_allen = (van_allen_data.data.minBy(_._1), van_allen_data.data.maxBy(_._1))

  val read_kp_data =
    omni_ops.omniFileToStream(omni_data.Quantities.Kp, Seq()) >
      omni_ops.processWithDateTime >
      IterableDataPipe((xs: (DateTime, Seq[Double])) => (xs._1, xs._2.head))

  val filter_kp_data = DataPipe[(DateTime, Double), Boolean](
    p => p._1.isAfter(time_limits_van_allen._1._1.minusHours(1)) && p._1.isBefore(time_limits_van_allen._2._1)
  )

  val process_time_stamp = DataPipe[DateTime, Int](d => {

    val period = new Duration(time_limits_van_allen._1._1, d)
    period.getStandardHours.toInt

  })

  val omni_file = pwd/'data/"omni2_2012.csv"

  val kp_data = dtfdata.dataset(Iterable(omni_file.toString()))
    .flatMap(read_kp_data)
    .filter(filter_kp_data)
    .map(process_time_stamp * Scaler[Double](_/10.0))



  timeLimits = (
    kp_data.data.minBy(_._1)._1,
    kp_data.data.maxBy(_._1)._1)

  val (tmin, tmax) = timeLimits

  lShellLimits = (
    van_allen_data.data.minBy(_._2._1)._2._1,
    van_allen_data.data.maxBy(_._2._1)._2._1
  )

  val kp_map = kp_data.data.toMap

  val scale_time = Scaler[Double](t => (t - tmin)/(tmax - tmin))
  val rescale_time = Scaler[Double](t => t*(tmax - tmin) + tmin)

  val compute_kp = DataPipe[Double, Double]((t: Double) => {
    if(t <= tmin) kp_map.minBy(_._1)._2
    else if(t >= tmax) kp_map.maxBy(_._1)._2
    else {
      val (lower, upper) = (kp_map(math.floor(t).toInt), kp_map(math.ceil(t).toInt))

      if(math.ceil(t) == math.floor(t)) lower
      else lower + (t - math.floor(t))*(upper - lower)/(math.ceil(t) - math.floor(t))
    }

  })

  val kp = rescale_time > compute_kp

  val psd_min = van_allen_data.data.minBy(_._2._2)._2._2

  val training_data = van_allen_data
    .map(process_time_stamp * (identityPipe[Double] * Scaler[Double](_/psd_min)))
    .map((p: (Int, (Double, Double))) => ((scale_time(p._1.toDouble), p._2._1), p._2._2))


  val chebyshev_hybrid_basis = HybridPSDBasis.chebyshev_laguerre_basis(
    lShellLimits, basisSize._1,
    timeLimits, basisSize._2)


  val seKernel = new GenExpSpaceTimeKernel[Double](
    1d, deltaL, deltaT)(
    sqNormDouble, l1NormDouble)

  val noiseKernel = new DiracTuple2Kernel(1.5)

  noiseKernel.block_all_hyper_parameters


  val model = new SGRadialDiffusionModel(
    Kp, dll_params,
    lambda_params,
    (0.01, 0.01d, 0.01, 0.01))(
    seKernel, noiseKernel,
    training_data,
    chebyshev_hybrid_basis,
    lShellLimits, timeLimits/*,
    hyper_param_basis = hyp_basis*/
  )

  val blocked_hyp = {
    model.blocked_hyper_parameters ++
      model.hyper_parameters.filter(c =>
        c.contains("dll") ||
          c.contains("base::") ||
          c.contains("lambda_")
      )
  }


  model.block(blocked_hyp:_*)


  val hyp = model.effective_hyper_parameters

  val h_prior = RDExperiment.hyper_prior(hyp)

  model.regCol = reg_galerkin
  model.regObs = reg_data

  //Create the MCMC sampler
  val mcmc_sampler = new AdaptiveHyperParameterMCMC[
    model.type, ContinuousDistr[Double]](
    model, h_prior, burn)


  //Draw samples from the posterior
  mcmc_sampler.iid(num_post_samples).draw


}