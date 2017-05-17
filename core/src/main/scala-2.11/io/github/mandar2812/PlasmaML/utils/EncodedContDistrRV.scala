package io.github.mandar2812.PlasmaML.utils

import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.pipes.Encoder
import io.github.mandar2812.dynaml.probability.ContinuousDistrRV

/**
  * Created by mandar on 16/05/2017.
  * */
class EncodedContDistrRV[Domain1, Domain2, Distr <: ContinuousDistr[Domain1]](
  base: Distr, encoder: Encoder[Domain1, Domain2])
  extends ContinuousDistrRV[Domain2] {

  override val underlyingDist = new ContinuousDistr[Domain2] {
    override def unnormalizedLogPdf(x: Domain2) = base.unnormalizedLogPdf(encoder.i(x))

    override def logNormalizer = base.logNormalizer

    override def draw() = encoder(base.draw())
  }
}

object EncodedContDistrRV {

  def apply[Domain1, Domain2, Distr <: ContinuousDistr[Domain1]](
    base: Distr, encoder: Encoder[Domain1, Domain2]): EncodedContDistrRV[Domain1, Domain2, Distr] =
    new EncodedContDistrRV(base, encoder)
}