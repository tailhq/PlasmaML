import _root_.io.github.mandar2812.dynaml.repl.Router.main
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data._
import ammonite.ops._
import org.joda.time._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SamePadding

@main
def main(test_year: Int = 2003) = {
  /*
* Mind your surroundings!
* */
  val os_name = System.getProperty("os.name")

  println("OS: "+os_name)

  val user_name = System.getProperty("user.name")

  println("Running as user: "+user_name)

  val home_dir_prefix = if(os_name.startsWith("Mac")) root/"Users" else root/'home

  val tempdir = home/"tmp"
  val tf_summary_dir = tempdir/("helios_goes_mdi_summaries_"+test_year)


  /*
  * Create a collated data set,
  * extract GOES flux data and join it
  * with eit195 (green filter) images.
  * */
  val data_dir = home_dir_prefix/user_name/"data_repo"/'helios
  val soho_dir = data_dir/'soho
  val goes_dir = data_dir/'goes

  val (year, month, day) = ("2003", "10", "28")

  val halloween_start = new DateTime(2003, 10, 28, 8, 0)

  val halloween_end = new DateTime(2003, 10, 29, 12, 0)


  val reduce_fn = (gr: Stream[(DateTime, (Double, Double))]) => {

    val max_flux = gr.map(_._2).max

    (gr.head._1, (math.log10(max_flux._1), math.log10(max_flux._2)))
  }

  val round_date = (d: DateTime) => {

    val num_minutes = 10

    val minutes: Int = d.getMinuteOfHour/num_minutes

    new DateTime(
      d.getYear, d.getMonthOfYear,
      d.getDayOfMonth, d.getHourOfDay,
      minutes*num_minutes)
  }

  val collated_data = helios.collate_data_range(
    new YearMonth(2001, 1), new YearMonth(2005, 12))(
    GOES(GOESData.Quantities.XRAY_FLUX_5m),
    goes_dir,
    goes_aggregation = 2,
    goes_reduce_func = reduce_fn,
    SOHO(SOHOData.Instruments.MDIMAG, 512),
    soho_dir,
    dt_round_off = round_date)



  val test_year_start = new DateTime(test_year, 1, 1, 0, 0)

  val test_year_end   = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition = (p: (DateTime, (Path, (Double, Double)))) =>
    if(p._1.isAfter(test_year_start) && p._1.isBefore(test_year_end)) false
    else true

  /*
  * After data has been joined/collated,
  * start loading it into tensors
  *
  * */

  val dataSet = helios.create_helios_data_set(
    collated_data,
    tt_partition,
    scaleDownFactor = 2)

  val trainImages = tf.data.TensorSlicesDataset(dataSet.trainData)

  val train_labels = dataSet.trainLabels(::, 0)

  val labels_mean = train_labels.mean()

  val labels_stddev = train_labels.subtract(labels_mean).square.mean().sqrt

  val trainLabels = tf.data.TensorSlicesDataset(train_labels.subtract(labels_mean).divide(labels_stddev))

  val trainData =
    trainImages.zip(trainLabels)
      .repeat()
      .shuffle(10000)
      .batch(64)
      .prefetch(10)

  /*
  * Start building tensorflow network/graph
  * */
  println("Building the regression model.")
  val input = tf.learn.Input(
    UINT8,
    Shape(
      -1,
      dataSet.trainData.shape(1),
      dataSet.trainData.shape(2),
      dataSet.trainData.shape(3))
  )

  val trainInput = tf.learn.Input(FLOAT32, Shape(-1))

  val layer = tf.learn.Cast("Input/Cast", FLOAT32) >>
    tf.learn.Conv2D("Conv2D_0", Shape(2, 2, 4, 64), 1, 1, SamePadding) >>
    tf.learn.AddBias(name = "Bias_0") >>
    tf.learn.ReLU("ReLU_0", 0.1f) >>
    tf.learn.Dropout("Dropout_0", 0.6f) >>
    tf.learn.Conv2D("Conv2D_1", Shape(2, 2, 64, 32), 2, 2, SamePadding) >>
    tf.learn.AddBias(name = "Bias_1") >>
    tf.learn.ReLU("ReLU_1", 0.1f) >>
    tf.learn.Dropout("Dropout_1", 0.6f) >>
    tf.learn.Conv2D("Conv2D_2", Shape(2, 2, 32, 16), 4, 4, SamePadding) >>
    tf.learn.AddBias(name = "Bias_2") >>
    tf.learn.ReLU("ReLU_2", 0.1f) >>
    tf.learn.Dropout("Dropout_2", 0.6f) >>
    tf.learn.Conv2D("Conv2D_3", Shape(2, 2, 16, 8), 8, 8, SamePadding) >>
    tf.learn.AddBias(name = "Bias_3") >>
    tf.learn.ReLU("ReLU_3", 0.1f) >>
    tf.learn.MaxPool("MaxPool_3", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
    tf.learn.Flatten("Flatten_3") >>
    tf.learn.Linear("FC_Layer_4", 128) >>
    tf.learn.ReLU("ReLU_4", 0.1f) >>
    tf.learn.Linear("FC_Layer_5", 64) >>
    tf.learn.ReLU("ReLU_5", 0.1f) >>
    tf.learn.Linear("FC_Layer_6", 8) >>
    tf.learn.Sigmoid("Sigmoid_6") >>
    tf.learn.Linear("OutputLayer", 1)

  val trainingInputLayer = tf.learn.Cast("TrainInput", INT64)
  val loss = tf.learn.L2Loss("Loss/L2") >> tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss", "ModelLoss")
  val optimizer = tf.train.AdaGrad(0.002)

  val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

  val (model, estimator) = tf.createWith(graph = Graph()) {
    val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    println("Training the linear regression model.")

    val estimator = tf.learn.FileBasedEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(100000)),
      Set(
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(5000)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(5000)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(5000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 5000))
    estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(0)))

    (model, estimator)
  }

  val accuracy = helios.calculate_rmse(dataSet.nTest, 4)(labels_mean, labels_stddev) _

  val testAccuracy = accuracy(
    dataSet.testData, dataSet.testLabels(::, 0))(
    (im: Tensor) => estimator.infer(() => im))

  print("Test accuracy = ")
  pprint.pprintln(testAccuracy)

}

