package io.github.mandar2812.PlasmaML.helios.core

import io.github.mandar2812.dynaml.utils.annotation.Experimental
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, Loss}
import org.platanios.tensorflow.api.ops.variables._

/**
  * <h3>Causal Dynamic time-lag Inference: Weighted Loss</h3>
  *
  * The loss functions assumes the prediction architecture
  * returns 2 &times; h number of outputs, where h = [[size_causal_window]].
  *
  * The first h outputs are predictions of the target signal, for
  * each time step in the horizon.
  *
  * The next h outputs are unscaled probability predictions regarding
  * the causal time lag between the input and output signals, computed
  * for any input pattern..
  *
  * The loss term for each data sample in the mini-batch is computed
  * independently as follows.
  *
  * L = &Sigma; <sub>i</sub> (|f<sub>i</sub> - y<sub>i</sub>|<sup>2</sup> &times; (1 + c &times; p<sub>i</sub>))
  * + &gamma; D(p<sup>*</sup>, p)
  *
  * @param name A string identifier for the loss instance.
  * @param size_causal_window The size of the finite time horizon
  *                           to look for causal effects.
  * @param prior_wt The multiplicative constant applied on the
  *                 "prior" term; i.e. a divergence term between the
  *                 predicted probability distribution of the
  *                 time lags and the so called
  *                 "target probability distribution".
  *
  * @param temperature The Gibbs temperature which scales the softmax probability
  *                    transformation while computing the target probability distribution.
  *
  * @param specificity Refers to the weight `c` in loss expression, a higher value of c
  *                    encourages the model to focus more on the learning input-output
  *                    relationships for the time steps which it perceives as more probable
  *                    to contain the causal time lag.
  *
  * @param divergence The kind of divergence term to be used as a prior over the probability
  *                   distribution predicted for the time lag. Available options include.
  *                   <ul>
  *                     <li>
  *                       <a href="https://en.wikipedia.org/wiki/Hellinger_distance">
  *                         Hellinger distance</a>
  *                     </li>
  *                     <li>
  *                       <a href="https://en.wikipedia.org/wiki/Kullback–Leibler_divergence">
  *                         Kullback-Leibler Divergence</a> (default)
  *                     </li>
  *                     <li>
  *                       <a href="https://en.wikipedia.org/wiki/Cross_entropy">Cross Entropy</a>
  *                     </li>
  *                     <li>
  *                       <a href="https://en.wikipedia.org/wiki/Jensen–Shannon_divergence">
  *                         Jensen-Shannon Divergence</a>
  *                     </li>
  *                   </ul>
  *
  * */
case class CausalDynamicTimeLag[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int,
  prior_wt: Double                                             = 1.5,
  error_wt: Double                                             = 1.0,
  temperature: Double                                          = 1.0,
  specificity: Double                                          = 1.0,
  divergence: CausalDynamicTimeLag.Divergence                  = CausalDynamicTimeLag.KullbackLeibler,
  target_distribution: CausalDynamicTimeLag.TargetDistribution = CausalDynamicTimeLag.Boltzmann) extends
  Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType: String = s"WTSLoss[horizon:$size_causal_window]"

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    val preds   = input._1._1
    val prob    = input._1._2
    val targets = input._2

    val model_errors = preds.subtract(targets.castTo[P])

    val target_prob = target_distribution match {
      case CausalDynamicTimeLag.Boltzmann => model_errors
        .square
        .multiply(Tensor(-1).toOutput.castTo[P])
        .divide(Tensor(temperature).castTo[P])
        .softmax()

      case CausalDynamicTimeLag.Uniform => dtf.tensor_f64(
        1, size_causal_window)(
        (1 to size_causal_window).map(_ => 1.0/size_causal_window):_*
      ).toOutput.castTo[P]

      case _ => model_errors
        .square
        .multiply(Tensor(-1).toOutput.castTo[P])
        .divide(Tensor(temperature).castTo[P])
        .softmax()
    }


    val prior_term = divergence(prob, target_prob)

    model_errors.square
      .multiply(prob.multiply(Tensor(specificity).castTo[P]).add(Tensor(1).toOutput.castTo[P]))
      .sum(axes = 1).mean()
      .multiply(Tensor(error_wt).toOutput.castTo[P])
      .add(prior_term.multiply(Tensor(prior_wt).castTo[P]))
      .reshape(Shape())
      .castTo[L]
  }
}

object CausalDynamicTimeLag {

  sealed trait Divergence {
    def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T]
  }

  object KullbackLeibler extends Divergence {

    override def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T] =
      p.divide(q).log.multiply(p).sum(axes = 1).mean[Int]()

    override def toString: String = "KullbackLeibler"
  }

  object JensenShannon extends Divergence {
    override def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T] = {

      val m = p.add(q).divide(Tensor(2).toOutput.castTo[T])

      (KullbackLeibler(p, m) + KullbackLeibler(q, m)).divide(Tensor(2).toOutput.castTo[T])

    }

    override def toString: String = "JensenShannon"
  }

  object CrossEntropy extends Divergence {
    override def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T] =
      q.multiply(p.log).sum(axes = 1).multiply(Tensor(-1).toOutput.castTo[T]).mean[Int]()

    override def toString: String = "CrossEntropy"
  }

  object Hellinger extends Divergence {
    override def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T] =
      q.sqrt.subtract(p.sqrt).square.sum(axes = 1).divide(Tensor(2).toOutput.castTo[T]).sqrt.mean[Int]()

    override def toString: String = "Hellinger"
  }

  object L2 extends Divergence {
    override def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T] =
      q.subtract(p).square.sum(axes = 1).divide(Tensor(2).toOutput.castTo[T]).mean[Int]()

    override def toString: String = "L2"
  }

  object Entropy extends Divergence {
    override def apply[T: TF: IsFloatOrDouble](p: Output[T], q: Output[T]): Output[T] =
      p.log.multiply(p).multiply(Tensor(-1).castTo[T]).sum(axes = 1).mean[Int]()

    override def toString: String = "ShannonEntropy"
  }


  sealed trait TargetDistribution

  object Boltzmann extends TargetDistribution {

    override def toString: String = "Boltzmann"
  }

  object Uniform extends TargetDistribution {

    override def toString: String = "Uniform"
  }

  def output_mapping[P: TF: IsFloatOrDouble](
    name: String,
    size_causal_window: Int): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {
      override val layerType: String = s"OutputWTSLoss[horizon:$size_causal_window]"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {
        (input(::, 0::size_causal_window), input(::, size_causal_window::).softmax())
      }
    }
}

case class CausalDynamicTimeLagI[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int) extends
  Loss[(Output[P], Output[T]), L](name) {

  override val layerType: String = s"L2Loss[horizon:$size_causal_window]"

  override def forwardWithoutContext(
    input: (Output[P], Output[T]))(
    implicit mode: Mode): Output[L] = {

    val preds    = input._1

    val targets  = input._2.castTo[P]

    preds.subtract(targets).square.sum(axes = 1).mean[Int]().castTo[L]
  }
}


case class CausalDynamicTimeLagII[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int,
  prior_wt: Double = 1.5,
  error_wt: Double = 1.0,
  temperature: Double = 1.0,
  specificity: Double = 1.0,
  divergence: CausalDynamicTimeLag.Divergence = CausalDynamicTimeLag.KullbackLeibler,
  target_distribution: CausalDynamicTimeLag.TargetDistribution = CausalDynamicTimeLag.Boltzmann) extends
  Loss[(Output[P], Output[P]), L](name) {

  override val layerType: String = s"CTL[horizon:$size_causal_window]"

  override def forwardWithoutContext(input: (Output[P], Output[P]))(implicit mode: Mode): Output[L] = {

    val prob    = input._1

    val model_errors = input._2

    val target_prob = target_distribution match {
      case CausalDynamicTimeLag.Boltzmann => model_errors
        .square
        .multiply(Tensor(-1).toOutput.castTo[P])
        .divide(Tensor(temperature).castTo[P])
        .softmax()

      case CausalDynamicTimeLag.Uniform => dtf.tensor_f64(
        1, size_causal_window)(
        (1 to size_causal_window).map(_ => 1.0/size_causal_window):_*
      ).toOutput.castTo[P]

      case _ => model_errors
        .square
        .multiply(Tensor(-1).toOutput.castTo[P])
        .divide(Tensor(temperature).castTo[P])
        .softmax()
    }


    val prior_term = divergence(prob, target_prob)

    model_errors
      .square
      .multiply(prob.multiply(Tensor(specificity).toOutput.castTo[P]))
      .sum(axes = 1)
      .mean[Int]()
      .multiply(Tensor(error_wt).toOutput.castTo[P])
      .add(prior_term.multiply(Tensor(prior_wt).toOutput.castTo[P]))
      .reshape(Shape())
      .castTo[L]
  }
}

object CausalDynamicTimeLagII {
  def output_mapping[P: TF: IsFloatOrDouble](name: String): Layer[Output[P], Output[P]] =
    new Layer[Output[P], Output[P]](name) {
      override val layerType: String = s"ProbCDT"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): Output[P] = {
        input.softmax()
      }
    }
}

@Experimental
case class CausalDynamicTimeLagSO[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int,
  prior_wt: Double = 1.5,
  error_wt: Double = 1.0,
  temperature: Double = 1.0,
  specificity: Double = 1.0,
  divergence: CausalDynamicTimeLag.Divergence = CausalDynamicTimeLag.KullbackLeibler) extends
  Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType: String = s"WTSLossSO[horizon:$size_causal_window]"

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    val preds               = input._1._1
    val repeated_preds      = tf.stack(Seq.fill(size_causal_window)(preds), axis = -1)
    val prob                = input._1._2
    val targets             = input._2.castTo[P]

    val model_errors = repeated_preds.subtract(targets)

    val target_prob = divergence match {
      case CausalDynamicTimeLag.Hellinger => model_errors
        .square.multiply(Tensor(-1).toOutput.castTo[P])
        .divide(Tensor(temperature).toOutput.castTo[P])
        .softmax()
      case _ => dtf.tensor_f64(
        1, size_causal_window)(
        (1 to size_causal_window).map(_ => 1.0/size_causal_window):_*).toOutput.castTo[P]
    }


    val prior_term = divergence(prob, target_prob)

    model_errors.square
      .multiply(prob.multiply(Tensor(specificity).toOutput.castTo[P]))
      .add(Tensor(1).toOutput.castTo[P])
      .sum(axes = 1).mean[Int]()
      .multiply(Tensor(error_wt).toOutput.castTo[P])
      .add(prior_term.multiply(Tensor(prior_wt).toOutput.castTo[P]))
      .reshape(Shape())
      .castTo[L]
  }
}


@Experimental
object WeightedTimeSeriesLossBeta {
  def output_mapping[P: TF: IsNotQuantized: IsFloatOrDouble](
    name: String,
    size_causal_window: Int): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {
      override val layerType: String = s"OutputWTSLossBeta[horizon:$size_causal_window]"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {
        val preds = input(::, 0::size_causal_window)
        val alpha_beta = input(::, size_causal_window::).square.add(Tensor(1.0).castTo[P])


        val (alpha, beta) = (alpha_beta(::, 0), alpha_beta(::, 1))

        val (stacked_alpha, stacked_beta) = (
          tf.stack(Seq.fill(size_causal_window)(alpha), axis = 1),
          tf.stack(Seq.fill(size_causal_window)(beta),  axis = 1))

        val index_times   = Tensor(0 until size_causal_window).reshape(Shape(1, size_causal_window)).toOutput.castTo[P]

        val n             = Tensor(size_causal_window - 1.0).toOutput.castTo[P]
        val n_minus_t     = index_times.castTo[P].multiply(Tensor(-1.0).toOutput.castTo[P]).add(n)

        val norm_const    = stacked_alpha.logGamma
          .add(stacked_beta.logGamma)
          .subtract(stacked_alpha.add(stacked_beta).logGamma)

        val prob          = index_times
          .castTo[P]
          .add(stacked_alpha).logGamma
          .add(n_minus_t.add(stacked_beta).logGamma)
          .add(n.add(Tensor(1.0).toOutput.castTo[P]).logGamma)
          .subtract(index_times.add(Tensor(1.0).toOutput.castTo[P]).logGamma)
          .subtract(n_minus_t.add(Tensor(1.0).toOutput.castTo[P]).logGamma)
          .subtract(norm_const).exp

        (preds, prob)
      }
    }
}

@Experimental
object WeightedTimeSeriesLossPoisson {
  def output_mapping[P: TF: IsFloatOrDouble](name: String, size_causal_window: Int): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {
      override val layerType: String = s"OutputPoissonWTSLoss[horizon:$size_causal_window]"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {

        val preds = input(::, 0::size_causal_window)

        val lambda = input(::, size_causal_window).softplus

        val stacked_lambda = tf.stack(Seq.fill(size_causal_window)(lambda), axis = 1)

        val index_times    = Tensor(0 until size_causal_window).reshape(Shape(1, size_causal_window)).castTo[P].toOutput

        val unsc_prob      = index_times.multiply(stacked_lambda.logGamma)
          .subtract(index_times.add(Tensor(1.0).castTo[P]).logGamma)
          .subtract(stacked_lambda)
          .exp

        val norm_const     = unsc_prob.sum(axes = 1)

        val prob           = unsc_prob.divide(tf.stack(Seq.fill(size_causal_window)(norm_const), axis = 1))

        (preds, prob)
      }
    }
}

@Experimental
object WeightedTimeSeriesLossGaussian {
  def output_mapping[P: TF: IsFloatOrDouble](name: String, size_causal_window: Int): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {
      override val layerType: String = s"OutputGaussianWTSLoss[horizon:$size_causal_window]"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {

        val preds = input(::, 0::size_causal_window)

        val mean = input(::, size_causal_window)

        //Precision is 1/sigma^2
        val precision = input(::, size_causal_window + 1).square

        val stacked_mean = tf.stack(Seq.fill(size_causal_window)(mean), axis = 1)
        val stacked_pre  = tf.stack(Seq.fill(size_causal_window)(precision), axis = 1)

        val index_times    = Tensor(0 until size_causal_window).reshape(Shape(1, size_causal_window)).castTo[P].toOutput

        val unsc_prob      = index_times.subtract(stacked_mean).square.multiply(Tensor(-0.5).castTo[P]).multiply(stacked_pre).exp

        val norm_const     = unsc_prob.sum(axes = 1)

        val prob           = unsc_prob.divide(tf.stack(Seq.fill(size_causal_window)(norm_const), axis = 1))

        (preds, prob)
      }
    }
}


object CausalDynamicTimeLagSO {
  def output_mapping[P: TF: IsFloatOrDouble](
    name: String,
    size_causal_window: Int): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {
      override val layerType: String = s"OutputWTSLossSO[horizon:$size_causal_window]"

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {
        (input(::, 0), input(::, 1::).softmax())
      }
    }
}


@Experimental
case class MOGrangerLoss[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int,
  error_exponent: Double = 2.0,
  weight_error: Double,
  scale_lags: Boolean = true) extends
  Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType = s"MOGrangerLoss[horizon:$size_causal_window]"

  private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    val alpha = Tensor(1.0).toOutput.castTo[P]
    val nu = Tensor(1.0).toOutput.castTo[P]
    val q = Tensor(1.0).toOutput.castTo[P]


    val predictions    = input._1._1
    val timelags       = input._1._2

    val targets        = input._2

    val repeated_times = tf.stack(Seq.fill(size_causal_window)(timelags), axis = -1)

    val index_times: Output[P] = Tensor(
      (0 until size_causal_window).map(_.toDouble)
    ).reshape(
      Shape(size_causal_window)
    ).toOutput.castTo[P]

    val error_tensor = predictions.subtract(targets.castTo[P])

    val convolution_kernel_temporal = error_tensor
      .abs.pow(Tensor(error_exponent).toOutput.castTo[P])
      .l2Normalize(axes = 1)
      .square
      .subtract(Tensor(1.0).toOutput.castTo[P])
      .multiply(Tensor(-1/2.0).toOutput.castTo[P])

    val weighted_temporal_loss_tensor = repeated_times
      .subtract(index_times)
      .square
      .multiply(convolution_kernel_temporal)
      .sum(axes = 1)
      .divide(convolution_kernel_temporal.sum(axes = 1))
      .mean[Int]()

    error_tensor
      .square
      .sum(axes = 1)
      .multiply(Tensor(0.5*weight_error).toOutput.castTo[P])
      .mean[Int]()
      .add(weighted_temporal_loss_tensor)
      .reshape(Shape())
      .castTo[L]
  }
}

object MOGrangerLoss {

  def output_mapping[P: TF: IsFloatOrDouble](
    name: String,
    size_causal_window: Int,
    scale_lags: Boolean = true): Layer[Output[P], (Output[P], Output[P])] =
    new Layer[Output[P], (Output[P], Output[P])](name) {

      override val layerType: String = s"OutputMOGrangerLoss[horizon:$size_causal_window]"

      private[this] val scaling = Tensor(size_causal_window.toDouble-1d)

      val alpha = Tensor(1.0).toOutput.castTo[P]
      val nu    = Tensor(1.0).toOutput.castTo[P]
      val q     = Tensor(1.0).toOutput.castTo[P]

      override def forwardWithoutContext(input: Output[P])(implicit mode: Mode): (Output[P], Output[P]) = {

        val lags = if (scale_lags) {
          input(::, -1)
            .multiply(alpha.add(Tensor(1E-6).toOutput.castTo[P]).square.multiply(Tensor(-1.0).toOutput.castTo[P]))
            .exp
            .multiply(q.square)
            .add(Tensor(1.0).toOutput.castTo[P])
            .pow(nu.square.pow(Tensor(-1.0).toOutput.castTo[P]).multiply(Tensor(-1.0).toOutput.castTo[P]))
            .multiply(Tensor(scaling).toOutput.castTo[P])
        } else {
          input(::, -1)
        }

        (input(::, 0::size_causal_window), lags)
      }
    }
}

case class ProbabilisticDynamicTimeLag[
P: TF: IsFloatOrDouble,
T: TF: IsNumeric,
L: TF : IsFloatOrDouble](
  override val name: String,
  size_causal_window: Int,
  temperature: Double = 1.0) extends
  Loss[((Output[P], Output[P]), Output[T]), L](name) {

  override val layerType: String = s"WTSLoss[horizon:$size_causal_window]"

  val divergence: CausalDynamicTimeLag.Divergence                  = CausalDynamicTimeLag.KullbackLeibler
  val target_distribution: CausalDynamicTimeLag.TargetDistribution = CausalDynamicTimeLag.Boltzmann

  override def forwardWithoutContext(
    input: ((Output[P], Output[P]), Output[T]))(
    implicit mode: Mode): Output[L] = {

    val preds   = input._1._1
    val prob    = input._1._2
    val targets = input._2

    val error_wt = tf.variable[P]("error_wt", Shape(), new RandomUniformInitializer)
    val specificity = tf.variable[P]("specificity", Shape(), new RandomUniformInitializer)

    val model_errors = preds.subtract(targets.castTo[P])

    val target_prob = model_errors
    .square
    .multiply(Tensor(-1).toOutput.castTo[P])
    .divide(Tensor(temperature).castTo[P])
    .softmax()


    val prior_term = divergence(prob, target_prob)

    model_errors.square
      .multiply(prob.multiply(specificity.square).add(Tensor(1).toOutput.castTo[P]))
      .sum(axes = 1).mean()
      .multiply(error_wt.square)
      .add(prior_term)
      .reshape(Shape())
      .castTo[L]
  }
}
