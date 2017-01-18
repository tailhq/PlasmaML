import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.{DiracKernel, PolynomialKernel}
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.pipes.{DataPipe, ParallelPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils.GaussianScaler
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}



DateTimeZone.setDefault(DateTimeZone.UTC)
val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")
val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H")

//Dump data file for each storm.
val stormsPipe =
  fileToStream > replaceWhiteSpaces > StreamDataPipe((stormEventData: String) => {
    val stormMetaFields = stormEventData.split(',')

    val eventId = stormMetaFields(0)
    val startDate = stormMetaFields(1)
    val startHour = stormMetaFields(2).take(2)

    val endDate = stormMetaFields(3)
    val endHour = stormMetaFields(4).take(2)

    val minDst = stormMetaFields(5).toDouble

    val stormCategory = stormMetaFields(6)

    val startStamp = formatter.parseDateTime(startDate+"/"+startHour).minusHours(6).getMillis/1000.0
    val endStamp = formatter.parseDateTime(endDate+"/"+endHour).getMillis/1000.0

    //For each storm dump the OMNI segment into a new csv file

    val dumpOMNI = fileToStream >
      replaceWhiteSpaces >
      StreamDataPipe((line: String) => {
        val stamp = dayofYearformat.parseDateTime(line.split(",").take(3).mkString("/")).getMillis/1000.0
        stamp >= startStamp && stamp <= endStamp
      }) > streamToFile("data/storm"+eventId+".csv")

    dumpOMNI("data/omni2_"+startDate.split("/").head+".csv")

  })

stormsPipe("data/geomagnetic_storms.csv")

//Dump data file for a given date-time range
val dataDump = StreamDataPipe((startAndEnd: ((String, String), String)) => {
  val startStamp = formatter.parseDateTime(startAndEnd._1._1).getMillis/1000.0
  val endStamp = formatter.parseDateTime(startAndEnd._1._2).getMillis/1000.0
  val dumpOMNI = (fileName: String) =>  fileToStream >
    replaceWhiteSpaces >
    StreamDataPipe((line: String) => {
      val stamp = dayofYearformat.parseDateTime(line.split(",").take(3).mkString("/")).getMillis/1000.0
      stamp >= startStamp && stamp <= endStamp
    }) > streamToFile("data/"+fileName)

  dumpOMNI(startAndEnd._2)("data/omni2_"+startAndEnd._1._1.split("/").head+".csv")

})

dataDump(
  Stream(
    (("2008/01/01/00", "2008/01/11/10"), "osa_training.csv"),
    (("2014/11/15/00", "2014/12/01/23"), "osa_validation.csv")))


//Dump the preprocessed training data used for the GP-AR model

val column = 40
val deltaT = 6

val processedDataDump = StreamDataPipe((startAndEnd: ((String, String), String)) => {
  val startStamp = formatter.parseDateTime(startAndEnd._1._1).getMillis/1000.0
  val endStamp = formatter.parseDateTime(startAndEnd._1._2).getMillis/1000.0
  val dumpOMNI = (fileName: String) =>
    fileToStream >
    replaceWhiteSpaces >
    StreamDataPipe((line: String) => {
      val stamp = dayofYearformat.parseDateTime(line.split(",").take(3).mkString("/")).getMillis/1000.0
      stamp >= startStamp && stamp <= endStamp
    }) >
    extractTrainingFeatures(
      List(0,1,2,column),
      Map(
        16 -> "999.9", 21 -> "999.9",
        24 -> "9999.", 23 -> "999.9",
        40 -> "99999", 22 -> "9999999.",
        25 -> "999.9", 28 -> "99.99",
        27 -> "9.999", 39 -> "999",
        45 -> "99999.99", 46 -> "99999.99",
        47 -> "99999.99")) >
      removeMissingLines >
      extractTimeSeries((year,day,hour) => (day * 24) + hour) >
      deltaOperation(deltaT, 0) >
      StreamDataPipe((record: (DenseVector[Double], Double)) => (record._1, DenseVector(record._2))) >
      gaussianScaling >
      DataPipe(
        DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) =>
          valuesToFile(fileName)(data.map(c => c._1.toArray.toSeq ++ c._2.toArray.toSeq))
        ),
        DataPipe((scaler: (GaussianScaler, GaussianScaler)) => {
          val means = scaler._1.mean.toArray.toSeq ++ scaler._2.mean.toArray.toSeq
          val sigmas = scaler._1.sigma.toArray.toSeq ++ scaler._2.sigma.toArray.toSeq
          val dump = Stream(means, sigmas)
          valuesToFile("scales_"+fileName)(dump)
        })
      )

  dumpOMNI(startAndEnd._2)("data/omni2_"+startAndEnd._1._1.split("/").head+".csv")

})


processedDataDump(Stream((("2008/01/01/00", "2008/01/11/10"), "osa_processed_training.csv")))

//Dump the Kernel matrix to a file

val kernel = new PolynomialKernel(1, 0.0) + new DiracKernel(1.173)

val kernelMatrixDump = DataPipe((startAndEnd: ((String, String), String)) => {
  val startStamp = formatter.parseDateTime(startAndEnd._1._1).getMillis/1000.0
  val endStamp = formatter.parseDateTime(startAndEnd._1._2).getMillis/1000.0

  val kernelMatPipe = (fileName: String) =>
    fileToStream >
      replaceWhiteSpaces >
      StreamDataPipe((line: String) => {
        val stamp = dayofYearformat.parseDateTime(line.split(",").take(3).mkString("/")).getMillis/1000.0
        stamp >= startStamp && stamp <= endStamp
      }) >
      extractTrainingFeatures(
        List(0,1,2,column),
        Map(
          16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")) >
      removeMissingLines >
      extractTimeSeries((year,day,hour) => (day * 24) + hour) >
      deltaOperation(deltaT, 0) >
      StreamDataPipe((record: (DenseVector[Double], Double)) => (record._1, DenseVector(record._2))) >
      gaussianScaling >
      DataPipe(
        (dataAndScales: (
          Stream[(DenseVector[Double], DenseVector[Double])],
            (GaussianScaler, GaussianScaler))) => {
          val kMat = kernel.buildKernelMatrix(dataAndScales._1.map(_._1), dataAndScales._1.length)
          breeze.linalg.csvwrite(new java.io.File(fileName), kMat.getKernelMatrix())
        })
  kernelMatPipe(startAndEnd._2)("data/omni2_"+startAndEnd._1._1.split("/").head+".csv")
})

kernelMatrixDump((("2008/01/01/00", "2008/01/11/10"), "osa_training_kernel_matrix.csv"))


val modelTestDump = DataPipe((startAndEnd: ((String, String), String)) => {

  val startStamp = formatter.parseDateTime("2008/01/01/00").getMillis / 1000.0
  val endStamp = formatter.parseDateTime("2008/01/11/10").getMillis / 1000.0

  val modelTrainPipe = fileToStream >
      replaceWhiteSpaces >
      StreamDataPipe((line: String) => {
        val stamp = dayofYearformat.parseDateTime(line.split(",").take(3).mkString("/")).getMillis/1000.0
        stamp >= startStamp && stamp <= endStamp
      }) >
      extractTrainingFeatures(
        List(0,1,2,column),
        Map(
          16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")) >
      removeMissingLines >
      extractTimeSeries((year,day,hour) => (day * 24) + hour) >
      deltaOperation(deltaT, 0) >
      StreamDataPipe((record: (DenseVector[Double], Double)) => (record._1, DenseVector(record._2))) >
      gaussianScaling >
      DataPipe(
        (dataAndScales: (
          Stream[(DenseVector[Double], DenseVector[Double])],
            (GaussianScaler, GaussianScaler))) => {

          val model = new GPRegression(
            new PolynomialKernel(1, 0.0),
            new DiracKernel(1.173),
            dataAndScales._1.map(c => (c._1, c._2(0))))
          (model, dataAndScales._2)
        })

  val (gp, scales) = modelTrainPipe("data/omni2_2008.csv")

  val stormStartStamp = formatter.parseDateTime(startAndEnd._1._1).minusHours(6).getMillis / 1000.0
  val stormEndStamp = formatter.parseDateTime(startAndEnd._1._2).getMillis / 1000.0

  val modelTestPipe = fileToStream >
    replaceWhiteSpaces >
    StreamDataPipe((line: String) => {
      val stamp = dayofYearformat.parseDateTime(line.split(",").take(3).mkString("/")).getMillis/1000.0
      stamp >= stormStartStamp && stamp <= stormEndStamp
    }) >
    extractTrainingFeatures(
      List(0,1,2,column),
      Map(
        16 -> "999.9", 21 -> "999.9",
        24 -> "9999.", 23 -> "999.9",
        40 -> "99999", 22 -> "9999999.",
        25 -> "999.9", 28 -> "99.99",
        27 -> "9.999", 39 -> "999",
        45 -> "99999.99", 46 -> "99999.99",
        47 -> "99999.99")) >
    removeMissingLines >
    extractTimeSeries((year,day,hour) => (day * 24) + hour) >
    deltaOperation(deltaT, 0) >
    StreamDataPipe((record: (DenseVector[Double], Double)) => (record._1, DenseVector(record._2))) >
    DataPipe((testData: Stream[(DenseVector[Double], DenseVector[Double])]) => {
      val scaledTestData = (scales._1*scales._2)(testData)
      val predictions = gp.test(scaledTestData.map(c => (c._1, c._2(0))))
      val res = (scales._2.i*scales._2.i)(predictions.map(c => (DenseVector(c._3), DenseVector(c._2))))

      res.map(c => (c._1.toArray++c._2.toArray).toSeq).toStream

    }) >
    valuesToFile(startAndEnd._2)

  modelTestPipe("data/omni2_"+startAndEnd._1._1.split("/").head+".csv")

})

modelTestDump((("1998/02/17/12", "1998/02/18/10"), "osa_storm1_preds.csv"))
