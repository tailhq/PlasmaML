import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data._
import ammonite.ops._
import org.joda.time._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SamePadding

/*
* Mind your surroundings!
* */
val os_name = System.getProperty("os.name")

println("OS: "+os_name)

val user_name = System.getProperty("user.name")

println("Running as user: "+user_name)

val home_dir_prefix = if(os_name.startsWith("Mac")) root/"Users" else root/'home

val tempdir = home/"tmp"
val tf_summary_dir = tempdir/"helios_goes_mdi_summaries"


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


val tt_partition = (p: (DateTime, (Path, (Double, Double)))) =>
  if(p._1.isAfter(halloween_start) && p._1.isBefore(halloween_end)) true
  else scala.util.Random.nextDouble() <= 0.7

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

val layer = tf.learn.Cast(FLOAT32) >>
  tf.learn.Conv2D(Shape(2, 2, 4, 64), 1, 1, SamePadding, name = "Conv2D_0") >>
  tf.learn.AddBias(name = "Bias_0") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.Dropout(0.6f) >>
  tf.learn.Conv2D(Shape(2, 2, 64, 32), 2, 2, SamePadding, name = "Conv2D_1") >>
  tf.learn.AddBias(name = "Bias_1") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.Dropout(0.6f) >>
  tf.learn.Conv2D(Shape(2, 2, 32, 16), 4, 4, SamePadding, name = "Conv2D_2") >>
  tf.learn.AddBias(name = "Bias_2") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.Dropout(0.6f) >>
  tf.learn.Conv2D(Shape(2, 2, 16, 8), 8, 8, SamePadding, name = "Conv2D_3") >>
  tf.learn.AddBias(name = "Bias_3") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.MaxPool(Seq(1, 2, 2, 1), 1, 1, SamePadding, name = "MaxPool_0") >>
  tf.learn.Flatten() >>
  tf.learn.Linear(128, name = "FC_Layer_0") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.Linear(64, name = "FC_Layer_1") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.Linear(8, name = "FC_Layer_2") >>
  tf.learn.Sigmoid() >>
  tf.learn.Linear(1, name = "OutputLayer")

val trainingInputLayer = tf.learn.Cast(INT64)
val loss = tf.learn.L2Loss() >> tf.learn.Mean() >> tf.learn.ScalarSummary("Loss")
val optimizer = tf.train.AdaGrad(0.002)

val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

val (model, estimator) = tf.createWith(graph = Graph()) {
  val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

  println("Training the linear regression model.")

  val estimator = tf.learn.InMemoryEstimator(
    model,
    tf.learn.Configuration(Some(summariesDir)),
    tf.learn.StopCriteria(maxSteps = Some(100000)),
    Set(
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(1000)),
      tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(1000)),
      tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
    tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 500))
  estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(10000)))

  (model, estimator)
}

def accuracy(images: Tensor, labels: Tensor): Float = {
  val predictions = estimator.infer(() => images)

  predictions
    .multiply(labels_stddev).add(labels_mean)
    .subtract(labels).cast(FLOAT32)
    .square.mean().scalar
    .asInstanceOf[Float]
}

val (trainAccuracy, testAccuracy) = (
  accuracy(dataSet.trainData, dataSet.trainLabels(::, 0)),
  accuracy(dataSet.testData, dataSet.testLabels(::, 0)))

print("Train accuracy = ")
pprint.pprintln(trainAccuracy)

print("Test accuracy = ")
pprint.pprintln(testAccuracy)


