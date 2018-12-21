/*
 * Copyright (c) 2016. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 * Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
 * Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
 * Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
 * Vestibulum commodo. Ut rhoncus gravida arcu.
 */

package io.github.mandar2812.PlasmaML.omni

import java.io.{BufferedWriter, File, FileWriter}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date, GregorianCalendar}

import breeze.linalg.DenseVector
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils
import io.github.mandar2812.dynaml.DynaMLPipe._
import org.apache.log4j.Logger
import org.joda.time.{DateTime, DateTimeZone}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

object Narmax extends DataPipe[DenseVector[Double], Double] {
  val narmax_params = DenseVector(
    0.8335, -3.083e-4, -6.608e-7, 0.13112,
    -2.1584e-10, 2.8405e-5, 1.5255e-10,
    7.3573e-5, 0.73433, 1.545e-4)

  override def run(data: DenseVector[Double]): Double = narmax_params dot data
}

/**
  * Created by mandar on 8/4/16.
  * */
object TestOmniNarmax {

  val logger = Logger.getLogger(this.getClass)

  DateTimeZone.setDefault(DateTimeZone.UTC)
  val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")
  val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H")

  var (dst_t_1, dst_t_2): (Double, Double) = (Double.NaN, Double.NaN)

  var (couplingFunc_t_1, couplingFunc_t_2, couplingFunc_t_3) = (Double.NaN, Double.NaN, Double.NaN)

  val processOmni = fileToStream >
    replaceWhiteSpaces >
    extractTrainingFeatures(
      List(0, 1, 2, 40, 24, 15, 16, 28),
      Map(
        16 -> "999.9", 16 -> "999.9",
        21 -> "999.9", 24 -> "9999.",
        23 -> "999.9", 40 -> "99999",
        22 -> "9999999.", 25 -> "999.9",
        28 -> "99.99", 27 -> "9.999", 39 -> "999",
        45 -> "99999.99", 46 -> "99999.99",
        47 -> "99999.99")) >
    removeMissingLines >
    extractTimeSeriesVec((year,day,hour) => {
      dayofYearformat.parseDateTime(year.toInt.toString+"/"+day.toInt.toString+"/"+hour.toInt.toString).getMillis/1000.0
    }) >
    StreamDataPipe((couple: (Double, DenseVector[Double])) => {
      val features = couple._2
      //Calculate the coupling function p^0.5 V^4/3 Bt sin^6(theta)
      val Bt = math.sqrt(math.pow(features(2), 2) + math.pow(features(3), 2))
      val sin_theta6 = math.pow(features(2)/Bt,6)
      val p = features(4)
      val v = features(1)
      val couplingFunc = math.sqrt(p)*math.pow(v, 4/3.0)*Bt*sin_theta6
      val Dst = features(0)
      (couple._1, DenseVector(Dst, couplingFunc))
    })

  def apply(start: String, end: String) = {

    val processDates: DataPipe[(String, String), Stream[(Double, DenseVector[Double])]] =
    DataPipe((limits: (String, String)) => (
      formatter.parseDateTime(limits._1).minusHours(4),
      formatter.parseDateTime(limits._2))) >
      DataPipe((dates: (DateTime, DateTime)) => "data/omni2_"+dates._1.getYear.toString+".csv") >
      processOmni

    val (trStampStart, trStampEnd) = (
      formatter.parseDateTime(start).minusHours(4).getMillis/1000.0,
      formatter.parseDateTime(end).getMillis/1000.0)


    val filterDataARX = StreamDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= trStampStart && couple._1 <= trStampEnd)

    val mpoPipe = processDates > filterDataARX > StreamDataPipe((couple: (Double, DenseVector[Double])) => {

      val finalFeatures = DenseVector(dst_t_1, couplingFunc_t_1, couplingFunc_t_1*dst_t_1,
        dst_t_2, math.pow(couplingFunc_t_2, 2.0),
        couplingFunc_t_3, math.pow(couplingFunc_t_1, 2.0),
        couplingFunc_t_2, 1.0, math.pow(dst_t_1, 2.0))

      val predicted = Narmax(finalFeatures)

      //update values
      dst_t_2 = dst_t_1

      if(dst_t_1.isNaN || predicted.isNaN) {
        dst_t_1 = couple._2(0)
      } else if (!predicted.isNaN) {
        dst_t_1 = predicted
      }

      couplingFunc_t_3 = couplingFunc_t_2
      couplingFunc_t_2 = couplingFunc_t_1
      couplingFunc_t_1 = couple._2(1)

      couple._1.toString+","+couple._2(0).toString+","+predicted.toString
    }) > streamToFile("data/NM_"+start.replaceAll("/", "_")+"_Res.csv")

    mpoPipe(start, end)
  }

  def apply(start: Int, end: Int) = {

    for(year <- start to end) {
      logger.info("Generating Narmax predictions for "+year)
      val omniFile = "data/omni2_"+year+".csv"
      val dataStream = processOmni(omniFile)

      val lines = dataStream.map((couple: (Double, DenseVector[Double])) => {

        val finalFeatures = DenseVector(dst_t_1, couplingFunc_t_1, couplingFunc_t_1*dst_t_1,
          dst_t_2, math.pow(couplingFunc_t_2, 2.0),
          couplingFunc_t_3, math.pow(couplingFunc_t_1, 2.0),
          couplingFunc_t_2, 1.0, math.pow(dst_t_1, 2.0))

        val predicted = Narmax(finalFeatures)

        //update values
        dst_t_2 = dst_t_1

        if(dst_t_1.isNaN || predicted.isNaN) {
          dst_t_1 = couple._2(0)
        } else if (!predicted.isNaN) {
          dst_t_1 = predicted
        }

        couplingFunc_t_3 = couplingFunc_t_2
        couplingFunc_t_2 = couplingFunc_t_1
        couplingFunc_t_1 = couple._2(1)

        couple._1.toString+","+couple._2(0).toString+","+predicted.toString
      })

      /*val pipe = processOmni  >
        StreamDataPipe((couple: (Double, DenseVector[Double])) => {

          val finalFeatures = DenseVector(dst_t_1, couplingFunc_t_1, couplingFunc_t_1*dst_t_1,
            dst_t_2, math.pow(couplingFunc_t_2, 2.0),
            couplingFunc_t_3, math.pow(couplingFunc_t_1, 2.0),
            couplingFunc_t_2, 1.0, math.pow(dst_t_1, 2.0))

          val predicted = Narmax(finalFeatures)

          //update values
          dst_t_2 = dst_t_1
          dst_t_1 = if(dst_t_1.isNaN) couple._2(0) else predicted
          couplingFunc_t_3 = couplingFunc_t_2
          couplingFunc_t_2 = couplingFunc_t_1
          couplingFunc_t_1 = couple._2(1)

          couple._1.toString+","+couple._2(0).toString+","+predicted.toString
        }) >
        streamToFile("data/NM_"+year+"_Res.csv")*/

      streamToFile("data/NM_"+year+"_Res.csv")(lines)
    }
  }

  def NMVariantExp = {

    DateTimeZone.setDefault(DateTimeZone.UTC)
    val format = DateTimeFormat.forPattern("yyyy/M/d")

    val writer =
      CSVWriter.open(
        new File("data/Omni_NM_Variant_StormsRes.csv"),
        append = true)

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val eventId = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val sJDate = format.parseDateTime(startDate)

          val startDay = sJDate.getDayOfYear
          val startHour = stormMetaFields(2).take(2)
          val startStamp = 24*startDay + startHour.toInt

          val endDate = stormMetaFields(3)
          val eJDate = format.parseDateTime(endDate)

          val endDay = eJDate.getDayOfYear
          val endHour = stormMetaFields(4).take(2)
          val endStamp = 24*endDay + endHour.toInt


          val minDst = stormMetaFields(5).toDouble

          val stormCategory = stormMetaFields(6)

          val year = startDate.split("/").head

          val NMFilePipe = fileToStream >
            StreamDataPipe((s: String) => s.split(",").map(_.toDouble)) >
            StreamDataPipe((line: Array[Double]) => line.head >= startStamp && line.head <= endStamp) >
            StreamDataPipe((line: Array[Double]) => (line.last, line(1))) >
            DataPipe((s: Stream[(Double, Double)]) => {
              val met = new RegressionMetrics(s.toList, s.length)

              val minPredDst = s.map(_._1).min
              val tp = s.map(_._1).indexOf(minPredDst)

              val minAct = s.map(_._2).min
              val ta = s.map(_._2).indexOf(minAct)
              /*"eventID","stormCat","order", "modelSize",
                "rmse", "corr", "deltaDstMin", "DstMin","deltaT"*/

              Seq(
                eventId, stormCategory, 1.0,
                0.0, met.rmse, met.corr,
                minPredDst - minDst,
                minDst, ta-tp
              )
            })

          writer.writeRow(NMFilePipe("data/NM_"+year+"_Res.csv"))
        })

    stormsPipe("data/geomagnetic_storms.csv")
    writer.close()
  }

  def apply(start: String = "2006/12/28/00",
            end: String = "2006/12/29/23",
            action: String = "test"): Seq[Seq[Double]] = {

    val names = Map(
      24 -> "Solar Wind Speed",
      16 -> "I.M.F Bz",
      40 -> "Dst",
      41 -> "AE",
      38 -> "Kp",
      39 -> "Sunspot Number",
      28 -> "Plasma Flow Pressure",
      23 -> "Proton Density"
    )

    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd/HH")
    val dateS: Date = sdf.parse(start)
    val dateE: Date = sdf.parse(end)

    val greg: GregorianCalendar = new GregorianCalendar()
    greg.setTime(dateS)
    val dayStart = greg.get(Calendar.DAY_OF_YEAR)
    val hourStart = greg.get(Calendar.HOUR_OF_DAY)
    val stampStart = (dayStart * 24) + hourStart
    val yearTest = greg.get(Calendar.YEAR)


    greg.setTime(dateE)
    val dayEnd = greg.get(Calendar.DAY_OF_YEAR)
    val hourEnd = greg.get(Calendar.HOUR_OF_DAY)
    val stampEnd = (dayEnd * 24) + hourEnd

    val preProcessPipe =
      fileToStream >
      replaceWhiteSpaces >
      extractTrainingFeatures(
        List(0, 1, 2, 40, 24, 15, 16, 28),
        Map(
          16 -> "999.9", 16 -> "999.9",
          21 -> "999.9", 24 -> "9999.",
          23 -> "999.9", 40 -> "99999",
          22 -> "9999999.", 25 -> "999.9",
          28 -> "99.99", 27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")) >
      removeMissingLines >
      extractTimeSeriesVec((year,day,hour) => (day * 24) + hour) >
      StreamDataPipe((couple: (Double, DenseVector[Double])) => {
        val features = couple._2
        //Calculate the coupling function p^0.5 V^4/3 Bt sin^6(theta)
        val Bt = math.sqrt(math.pow(features(2), 2) + math.pow(features(3), 2))
        val sin_theta6 = math.pow(features(2)/Bt,6)
        val p = features(4)
        val v = features(1)
        val couplingFunc = math.sqrt(p)*math.pow(v, 4/3.0)*Bt*sin_theta6
        val Dst = features(0)
        (couple._1, DenseVector(Dst, couplingFunc))
      }) > StreamDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= stampStart && couple._1 <= stampEnd) >
      deltaOperationARX(List(2, 3)) >
      StreamDataPipe((couple: (DenseVector[Double], Double)) => {
        val vec = couple._1
        val Dst_t_1 = vec(0)
        val Dst_t_2 = vec(1)

        val couplingFunc_t_1 = vec(2)
        val couplingFunc_t_2 = vec(3)
        val couplingFunc_t_3 = vec(4)

        val finalFeatures = DenseVector(Dst_t_1, couplingFunc_t_1, couplingFunc_t_1*Dst_t_1,
          Dst_t_2, math.pow(couplingFunc_t_2, 2.0),
          couplingFunc_t_3, math.pow(couplingFunc_t_1, 2.0),
          couplingFunc_t_2, 1.0, math.pow(Dst_t_1, 2.0))


        (Narmax(finalFeatures), couple._2)
      }) > DataPipe((scoresAndLabels: Stream[(Double, Double)]) => {

        val metrics = new RegressionMetrics(scoresAndLabels.toList,
          scoresAndLabels.length)


        metrics.print()
        metrics.generatePlots()

        //Plotting time series prediction comparisons
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        legend(List("Time Series", "Predicted Time Series (one hour ahead)"))
        unhold()


        logger.info("Printing One Step Ahead (OSA) Performance Metrics")
        metrics.print()
        val (timeObs, timeModel, peakValuePred, peakValueAct) = names(40) match {
          case "Dst" =>
            (scoresAndLabels.map(_._2).zipWithIndex.min._2,
              scoresAndLabels.map(_._1).zipWithIndex.min._2,
              scoresAndLabels.map(_._1).min,
              scoresAndLabels.map(_._2).min)
          case _ =>
            (scoresAndLabels.map(_._2).zipWithIndex.max._2,
              scoresAndLabels.map(_._1).zipWithIndex.max._2,
              scoresAndLabels.map(_._1).max,
              scoresAndLabels.map(_._2).max)
        }

        logger.info("Timing Error; OSA Prediction: "+(timeObs-timeModel))

        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        unhold()


        action match {
          case "test" =>
            Seq(
              Seq(yearTest.toDouble, 1.0, scoresAndLabels.length.toDouble,
                metrics.mae, metrics.rmse, metrics.Rsq,
                metrics.corr, metrics.modelYield,
                timeObs.toDouble - timeModel.toDouble,
                peakValuePred,
                peakValueAct)
            )
          case "predict" =>
            scoresAndLabels.map(c => Seq(c._2, c._1))
        }


      })


    preProcessPipe.run("data/omni2_"+yearTest+".csv")
  }
}


object TestOmniTL {

  val logger = Logger.getLogger(this.getClass)

  def apply(start: String = "2006/12/28/00",
            end: String = "2006/12/29/23",
            action: String = "test") = {

    val names = Map(
      24 -> "Solar Wind Speed",
      16 -> "I.M.F Bz",
      40 -> "Dst",
      41 -> "AE",
      38 -> "Kp",
      39 -> "Sunspot Number",
      28 -> "Plasma Flow Pressure",
      23 -> "Proton Density"
    )

    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd/HH")
    val dateS: Date = sdf.parse(start)
    val dateE: Date = sdf.parse(end)

    val greg: GregorianCalendar = new GregorianCalendar()
    greg.setTime(dateS)
    val dayStart = greg.get(Calendar.DAY_OF_YEAR)
    val hourStart = greg.get(Calendar.HOUR_OF_DAY)
    val stampStart = (dayStart * 24) + hourStart


    val yearStart = greg.get(Calendar.YEAR).toString

    val monthStart = if(greg.get(Calendar.MONTH) < 9) {
      "0"+(greg.get(Calendar.MONTH)+1).toString
    } else {
      (greg.get(Calendar.MONTH)+1).toString
    }

    val fileNameS = "dst_"+yearStart+"_"+monthStart+".txt"

    greg.setTime(dateE)
    val dayEnd = greg.get(Calendar.DAY_OF_YEAR)
    val hourEnd = greg.get(Calendar.HOUR_OF_DAY)
    val stampEnd = (dayEnd * 24) + hourEnd

    val yearEnd = greg.get(Calendar.YEAR).toString

    val monthEnd = if(greg.get(Calendar.MONTH) < 9) {
      "0"+(greg.get(Calendar.MONTH)+1).toString
    } else {
      (greg.get(Calendar.MONTH)+1).toString
    }

    val fileNameE = "dst_"+yearEnd+"_"+monthEnd+".txt"


    // Create two pipes pipeTL and pipeDst


    val pipeDst = fileToStream >
      replaceWhiteSpaces > extractTrainingFeatures(
      List(0, 1, 2, 40),
      Map(
        16 -> "999.9", 16 -> "999.9",
        21 -> "999.9", 24 -> "9999.",
        23 -> "999.9", 40 -> "99999",
        22 -> "9999999.", 25 -> "999.9",
        28 -> "99.99", 27 -> "9.999", 39 -> "999",
        45 -> "99999.99", 46 -> "99999.99",
        47 -> "99999.99")) >
      removeMissingLines >
      extractTimeSeries((year,day,hour) => (day * 24) + hour) >
      StreamDataPipe((couple: (Double, Double)) =>
      couple._1 >= stampStart && couple._1 <= stampEnd) >
      StreamDataPipe((couple: (Double, Double)) => {
        couple._2
      })


    val preprocessTLFile = fileToStream >
      dropHead >
      replaceWhiteSpaces >
      StreamDataPipe((line: String) => {
        //Split line using comma
        val spl = line.split(",")
        val datetime = spl.head
        val datespl = datetime.split("-")

        val hour = datespl.last.split(":").head.toDouble
        val dayNum = datespl.head.split("/").last.toDouble

        val dst = spl.last.toDouble
        (dayNum*24 + hour,dst)
      }) >
      StreamDataPipe((couple: (Double, Double)) =>
        couple._1 >= stampStart && couple._1 <= stampEnd) >
      DataPipe((tlData: Stream[(Double, Double)]) => {
        tlData.grouped(6).toStream.map{gr => (gr.head._1, gr.map(_._2).sum/gr.length.toDouble)}
      })

    val actualDst = pipeDst.run("data/omni2_"+yearStart+".csv")

    val tlDstPrediction = if(fileNameS == fileNameE) {
      val pipeTL = preprocessTLFile > StreamDataPipe((couple: (Double, Double)) => couple._2)
      pipeTL.run("data/"+fileNameS)
    } else {
      val pipeTL = DataPipe(preprocessTLFile, preprocessTLFile) >
        DataPipe((couple: (Stream[(Double, Double)], Stream[(Double, Double)])) => {
          couple._1 ++ couple._2
        }) >
        StreamDataPipe((couple: (Double, Double)) => couple._2)
      pipeTL.run(("data/"+fileNameS, "data/"+fileNameE))
    }

    val scoresAndLabels = tlDstPrediction zip actualDst

    val metrics = new RegressionMetrics(scoresAndLabels.toList,
      scoresAndLabels.length)


    metrics.print()
    metrics.generatePlots()

    //Plotting time series prediction comparisons
    line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
    hold()
    line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
    legend(List("Time Series", "Predicted Time Series (one hour ahead)"))
    unhold()


    logger.info("Printing One Step Ahead (OSA) Performance Metrics")
    metrics.print()
    val (timeObs, timeModel, peakValuePred, peakValueAct) = names(40) match {
      case "Dst" =>
        (scoresAndLabels.map(_._2).zipWithIndex.min._2,
          scoresAndLabels.map(_._1).zipWithIndex.min._2,
          scoresAndLabels.map(_._1).min,
          scoresAndLabels.map(_._2).min)
      case _ =>
        (scoresAndLabels.map(_._2).zipWithIndex.max._2,
          scoresAndLabels.map(_._1).zipWithIndex.max._2,
          scoresAndLabels.map(_._1).max,
          scoresAndLabels.map(_._2).max)
    }

    logger.info("Timing Error; OSA Prediction: "+(timeObs-timeModel))

    line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
    hold()
    line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
    unhold()

    action match {
      case "test" =>
        Seq(
          Seq(yearStart.toDouble, 1.0, scoresAndLabels.length.toDouble,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield,
            timeObs.toDouble - timeModel.toDouble,
            peakValuePred,
            peakValueAct)
        )
      case "predict" =>
        scoresAndLabels.map(c => Seq(c._2, c._1))
    }


  }

  def prepareTLFiles() = {

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val startDate = stormMetaFields(1)
          val endDate = stormMetaFields(3)

          val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd")
          val greg: GregorianCalendar = new GregorianCalendar()

          greg.setTime(sdf.parse(startDate))
          val yearStart = greg.get(Calendar.YEAR).toString

          val monthStart = if(greg.get(Calendar.MONTH) < 9) {
            "0"+(greg.get(Calendar.MONTH)+1).toString
          } else {
            (greg.get(Calendar.MONTH)+1).toString
          }

          val fileNameS = "dst_"+yearStart+"_"+monthStart+".txt"

          greg.setTime(sdf.parse(endDate))

          val yearEnd = greg.get(Calendar.YEAR).toString

          val monthEnd = if(greg.get(Calendar.MONTH) < 9) {
            "0"+(greg.get(Calendar.MONTH)+1).toString
          } else {
            (greg.get(Calendar.MONTH)+1).toString
          }

          val fileNameE = "dst_"+yearEnd+"_"+monthEnd+".txt"

          if(fileNameE == fileNameS) {
            logger.info("Same Month")
            utils.downloadURL(
              "http://lasp.colorado.edu/space_weather/dsttemerin/archive/"+fileNameS,
              "data/"+fileNameS)
          } else {
            logger.info("Different Months!")
            utils.downloadURL(
              "http://lasp.colorado.edu/space_weather/dsttemerin/archive/"+fileNameS,
              "data/"+fileNameS)

            utils.downloadURL(
              "http://lasp.colorado.edu/space_weather/dsttemerin/archive/"+fileNameE,
              "data/"+fileNameE)
          }

        })

    stormsPipe.run("data/geomagnetic_storms.csv")

  }

}


object DstNMTLExperiment {

  def apply(model: String = "NM", action:String = "test", fileID: String = "") = {
    val writer =
      CSVWriter.open(
        new File("data/Omni"+model+fileID+"StormsRes.csv"),
        append = true)

    val stormsPipe =
      fileToStream >
        replaceWhiteSpaces >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val eventId = stormMetaFields(0)
          val startDate = stormMetaFields(1)
          val startHour = stormMetaFields(2).take(2)

          val endDate = stormMetaFields(3)
          val endHour = stormMetaFields(4).take(2)

          val minDst = stormMetaFields(5).toDouble

          val stormCategory = stormMetaFields(6)

          val res = model match {
            case "NM" => TestOmniNarmax(
              startDate+"/"+startHour,
              endDate+"/"+endHour,
              action = action)

            case "TL" => TestOmniTL(
              startDate+"/"+startHour,
              endDate+"/"+endHour,
              action = action)
          }

          if(action == "test") {
            val row = Seq(
              eventId, stormCategory, 1.0,
              0.0, res.head(4), res.head(6),
              res.head(9)-res.head(10),
              res.head(10), res.head(8)
            )

            writer.writeRow(row)
          } else {
            writer.writeAll(res)
          }

        })

    stormsPipe.run("data/geomagnetic_storms.csv")
  }
}