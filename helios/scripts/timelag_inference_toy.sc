import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.{DynaMLPipe => Pipe}
import _root_.io.github.mandar2812.dynaml.pipes._
import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.dynaml.probability.RandomVariable
import breeze.linalg.{DenseMatrix, qr}
import breeze.stats.distributions.{Gamma, Gaussian, LogNormal, Uniform}
import com.quantifind.charts.Highcharts._
import _root_.io.github.mandar2812.dynaml.probability.distributions.TruncatedGaussian
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.PlasmaML.utils


@main
def main(d: Int = 3, n: Int = 5, noise: Double = 0.5) = {

  val random_gaussian_vec = DataPipe((i: Int) => RandomVariable(
    () => dtf.tensor_f32(i, 1)((0 until i).map(_ => scala.util.Random.nextGaussian()*noise):_*)
  ))

  val normalise = DataPipe((t: RandomVariable[Tensor]) => t.draw) //.l2Normalize(0))

  val normalised_gaussian_vec = random_gaussian_vec > normalise

  val x0 = normalised_gaussian_vec(d)

  val principal_direction = normalised_gaussian_vec(d)

  val random_gaussian_mat = DataPipe(
    (n: Int) => DenseMatrix.rand(n, n, Gaussian(0d, noise))
  )

  val random_diag_mat = MetaPipe(
    (rv: RandomVariable[Double]) => (n: Int) =>
      RandomVariable(() => tf.diag(dtf.tensor_f32(n)(rv.iid(n).draw:_*)).evaluate())
  )

  //val rand_diag_ln_mat = random_diag_mat(RandomVariable(LogNormal(0d, 0.25)))

  val rand_diag_trnorm_mat = random_diag_mat(RandomVariable(TruncatedGaussian(1d, noise, 0.5d, 1.5d)))

  val rand_rot_mat =
    random_gaussian_mat >
      DataPipe((m: DenseMatrix[Double]) => qr(m).q) >
      DataPipe((m: DenseMatrix[Double]) => dtf.tensor_f32(m.rows, m.rows)(m.toArray:_*).transpose())


  val rotation = rand_rot_mat(d)

  val get_rotation_operator = MetaPipe((rotation_mat: Tensor) => (x: Tensor) => rotation_mat.matmul(x))

  val rotation_op = get_rotation_operator(rotation)

  val scaling_op = DataPipe2((scale: Tensor, x: Tensor) => scale.matmul(x))

  val translation_op = DataPipe2((tr: Tensor, x: Tensor) => tr.add(x))

  val scaling_mats = rand_diag_trnorm_mat(d).iid(n).draw

  val translation_vecs = random_gaussian_vec(d).iid(n-1).draw

  val x_tail = translation_vecs.scanLeft(x0)((x, sc) => translation_op(sc, rotation_op(x)))
    //scaling_mats.scanLeft(x0)((x, sc) => scaling_op(sc, rotation_op(x)))



  val x: Seq[Tensor] = Stream(x0) ++ x_tail

  val velocity_pipe = DataPipe((v: Tensor) => v.square.sum().sqrt.scalar.asInstanceOf[Float])

  val (lower, upper) = (2, 10)

  def id[T] = Pipe.identityPipe[T]

  val calculate_outputs =
    velocity_pipe >
    BifurcationPipe(
      DataPipe((v: Float) => 10/(v+ 1E-6)), //utils.generalized_logistic(lower, upper, 10d, 0, 1.5, lower, 1d)(1d/(v + 1E-5))),
      id[Float]) >
    DataPipe(DataPipe((d: Double) => d.toInt), id[Float])



  val generate_data = StreamDataPipe(
    DataPipe(id[Int], BifurcationPipe(id[Tensor], calculate_outputs))  >
    DataPipe((pattern: (Int, (Tensor, (Int, Float)))) =>
      ((pattern._1, pattern._2._1), (pattern._1+pattern._2._2._1, pattern._2._2._2)))
  )





  val times = (0 until n).toStream

  val data = generate_data(times.zip(x))

  val energies = data.map(_._2._2)

  //val x_proj = x.map(v => v.transpose().matmul(principal_direction))

  x.foreach(v => println(v.summarize()+"\n"))

  x.foreach(v => print(v.transpose().matmul(v).scalar+"  "))

  spline(energies)
  title("Energy Time Series")

  val effect_times = data.map(_._2._1)

  spline(effect_times)
  hold()
  spline(data.map(_._1._1))
  title("Time Warping/Delay")
  xAxis("Time of Cause, t")
  yAxis("Time of Effect, "+0x03C6.toChar+"(t)")
  legend(Seq("t_ = "+0x03C6.toChar+"(t)", "t_ = t"))
  unhold()
}
