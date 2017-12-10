import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data._
import ammonite.ops._
import com.sksamuel.scrimage.Image
import org.joda.time._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SamePadding


val data_dir_name = "data_repo"//"tmp"//

val data_dir = home/data_dir_name///'helios
val soho_dir = data_dir/'soho
val goes_dir = data_dir/'goes

val tempdir = home/"tmp"

val (year, month, day) = ("2003", "10", "28")

val halloween_start = new DateTime(2003, 10, 28, 8, 59)

val reduce_fn = (gr: Stream[(DateTime, (Double, Double))]) => {

  val max_flux = gr.map(_._2).max

  (gr.head._1, (math.log10(max_flux._1), math.log10(max_flux._2)))
}

val round_date = (d: DateTime) => {
  val minutes: Int = d.getMinuteOfHour/10

  new DateTime(
    d.getYear, d.getMonthOfYear,
    d.getDayOfMonth, d.getHourOfDay,
    minutes*10)
}

val collated_data = helios.collate_data(
  new YearMonth(year.toInt, month.toInt))(
  GOES(GOESData.Quantities.XRAY_FLUX_5m),
  goes_dir,
  goes_aggregation = 2,
  goes_reduce_func = reduce_fn,
  SOHO(SOHOData.Instruments.EIT195, 512),
  soho_dir,
  dt_round_off = round_date)

val dataSet = helios.create_helios_data_set(collated_data, _ => scala.util.Random.nextDouble() <= 0.8)

val trainImages = tf.data.TensorSlicesDataset(dataSet.trainData)

val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels(::, 0))

val trainData =
  trainImages.zip(trainLabels)
    .repeat()
    .shuffle(10000)
    .batch(32)
    .prefetch(10)

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
  tf.learn.Conv2D(Shape(2, 2, 4, 16), 1, 1, SamePadding, name = "Conv2D_0") >>
  tf.learn.AddBias(name = "Bias_0") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.MaxPool(Seq(1, 2, 2, 1), 1, 1, SamePadding, name = "MaxPool_0") >>
  tf.learn.Conv2D(Shape(2, 2, 16, 32), 1, 1, SamePadding, name = "Conv2D_1") >>
  tf.learn.AddBias(name = "Bias_1") >>
  tf.learn.ReLU(0.1f) >>
  tf.learn.MaxPool(Seq(1, 2, 2, 1), 1, 1, SamePadding, name = "MaxPool_1") >>
  tf.learn.Flatten() >>
  tf.learn.Linear(256, name = "Layer_2") >> tf.learn.ReLU(0.1f) >>
  tf.learn.Linear(1, name = "OutputLayer")

val trainingInputLayer = tf.learn.Cast(INT64)
val loss = tf.learn.L2Loss() >> tf.learn.Mean() >> tf.learn.ScalarSummary("Loss")
val optimizer = tf.train.AdaGrad(0.1)

val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

println("Training the linear regression model.")
val summariesDir = java.nio.file.Paths.get((tempdir/"helios_goes_soho_summaries").toString())
val estimator = tf.learn.FileBasedEstimator(
  model,
  tf.learn.Configuration(Some(summariesDir)),
  tf.learn.StopCriteria(maxSteps = Some(100000)),
  Set(
    tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
    tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
    tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(100))),
  tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 100))
estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(500)))

def accuracy(images: Tensor, labels: Tensor): Float = {
  val predictions = estimator.infer(() => images)
  predictions.subtract(labels).cast(FLOAT32).square.mean().scalar.asInstanceOf[Float]
}

val (trainAccuracy, testAccuracy) = (
  accuracy(dataSet.trainData, dataSet.trainLabels),
  accuracy(dataSet.testData, dataSet.testLabels))

print("Train accuracy = ")
pprint.pprintln(trainAccuracy)

print("Test accuracy = ")
pprint.pprintln(testAccuracy)


