{
  import breeze.linalg._
  import breeze.numerics.Bessel
  import com.quantifind.charts.Highcharts._

  import io.github.mandar2812.dynaml.pipes.DataPipe
  import io.github.mandar2812.dynaml.probability._

  import io.github.mandar2812.PlasmaML.dynamics.diffusion._

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

  val omega = 2*math.Pi/(lShellLimits._2 - lShellLimits._1)

  val initialPSD = (l: Double) => Bessel.i1(omega*(l - lShellLimits._1))*1E2

  line(timeVec.map(t => (t, Kp(t))).toSeq)
  xAxis("time")
  yAxis("Kp")
  title("Evolution of Kp")

  spline(lShellVec.map(l => (l, initialPSD(l))).toSeq)
  xAxis("L")
  yAxis("f(L, 0)")
  title("Phase Space Density Profile, t = 0")

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

  (1 to 15).map(_.toDouble).foreach(value => {

    //Diffusion Field
    val dll_alpha = 1d
    val dll_beta = 10d
    val dll_a = -9.325
    val dll_b = 0.506

    //Loss Process
    val lambda_alpha = math.pow(10d, -4)/2.4
    val lambda_beta = 1d
    val lambda_a = 2.5d
    val lambda_b = 0.18

    val (q_alpha, q_beta, q_a, q_b) = (0.1, value, -0.12, 0.3)

    //Create ground truth diffusion parameter functions
    val dll = (l: Double, t: Double) => dll_alpha*math.pow(l, dll_beta)*math.pow(10, dll_a + dll_b*Kp(t))

    val Q = (l: Double, t: Double) => q_alpha*math.pow(l, q_beta)*math.pow(10, q_a + q_b*Kp(t))

    val lambda = (l: Double, t: Double) => lambda_alpha*math.pow(l, lambda_beta)*math.pow(10, lambda_a + lambda_b*Kp(t))

    //Create ground truth PSD data and corrupt it with statistical noise.
    val groundTruth = rds.solve(Q, dll, lambda)(initialPSD)
    val ground_truth_matrix = DenseMatrix.horzcat(groundTruth.map(_.asDenseMatrix.t):_*)
    val measurement_noise = GaussianRV(0.0, 0.5)

    val noise_mat = DenseMatrix.tabulate[Double](nL+1, nT+1)((_, _) => measurement_noise.draw)
    val data: DenseMatrix[Double] = ground_truth_matrix + noise_mat


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
    title("Evolution of Phase Space Density f(L,t) "+0x03B2.toChar+" = "+value)
    xAxis("time")
    yAxis("f(L,t)")


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
      .toSeq.map(s => "t = "+"%3f".format(s)))

    title("Variation of Phase Space Density f(L,t) "+0x03B2.toChar+" = "+value)
    xAxis("L")
    yAxis("f(L,t)")


  })


}
