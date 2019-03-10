package io.github.mandar2812.PlasmaML.helios.core

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * <h3>Compression &amp; Representation</h3>
  *
  * A layer which consists of compression and reconstruction modules.
  * Useful for unsupervised learning tasks.
  * */
case class AutoEncoder[I, J](
  override val name: String,
  encoder: Layer[I, J],
  decoder: Layer[J, I]) extends
  Layer[I, (J, I)](name) {

  self =>

  override val layerType: String = s"Representation[${encoder.layerType}, ${decoder.layerType}]"

  override def forwardWithoutContext(input: I)(implicit mode: Mode): (J, I) = {

    val encoding = encoder(input)

    val reconstruction = decoder(encoding)

    (encoding, reconstruction)
  }

  def >>[S](other: AutoEncoder[J, S]): ComposedAutoEncoder[I, J, S] = new ComposedAutoEncoder(name, self, other)
}

class ComposedAutoEncoder[I, J, K](
  override val name: String,
  encoder1: AutoEncoder[I, J],
  encoder2: AutoEncoder[J, K]) extends
  AutoEncoder[I, K](
    s"Compose[${encoder1.name}, ${encoder2.name}]",
    encoder1.encoder >> encoder2.encoder,
    encoder2.decoder >> encoder1.decoder)