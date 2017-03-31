import breeze.linalg.DenseVector
import com.sksamuel.scrimage._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels.{DiracKernel, GaussianSpectralKernel, MAKernel}
import io.github.mandar2812.dynaml.models.bayes.LinearTrendGaussianPrior

import scala.collection.mutable.{MutableList => MutList}

//Load the image into an object, and scale it by 0.25
val im = Image.fromFile(new java.io.File("../../tmp/20011030_0135_mdimag_512.jpg")).scale(0.0625)


val scale = 512*0.0625
val imageData: MutList[(DenseVector[Double], Double)] = MutList()

im.foreach((x, y, p) => {
  val coordinates: DenseVector[Double] = DenseVector(x/scale, y/scale)
  val pixelInt = PixelTools.gray(p.toARGBInt)
  imageData += ((coordinates, pixelInt))
})

val encoder = GaussianSpectralKernel.getEncoderforBreezeDV(2)
implicit val field = VectorField(2)

val n = new DiracKernel(2.5)
val n1 = new MAKernel(1.5)

val gsmKernel = GaussianSpectralKernel[DenseVector[Double]](
  DenseVector(0.5, 0.5), DenseVector(1.0, 1.0),
  encoder)


val gp_prior = new LinearTrendGaussianPrior[DenseVector[Double]](gsmKernel, n, DenseVector.zeros[Double](2), 128.0)


gp_prior.globalOptConfig_(Map("gridStep" -> "0.15", "gridSize" -> "2", "globalOpt" -> "GPC", "policy" -> "GS"))
val gpModel = gp_prior.posteriorModel(imageData)

