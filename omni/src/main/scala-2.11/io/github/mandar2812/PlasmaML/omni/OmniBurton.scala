package io.github.mandar2812.PlasmaML.omni

import io.github.mandar2812.PlasmaML.omni.OmniOSA._
import io.github.mandar2812.dynaml.DynaMLPipe.extractTimeSeries
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}

/**
  * @author mandar2812 30/03/2017.
  *
  * Contains work-flows for running experiments
  * on hybrid GP-Burton injection inference and
  * prediction models.
  *
  * The basic premise of this code is the Burton ODE
  * for time evolution of the Dst index
  *
  * d Dst(t)/dt + lambda*Dst(t) = Q(t)
  *
  * Q(t) ~ GP(0, K(t, t'))
  */
object OmniBurton {

  /**
    * @return A [[DataPipe]] that processes date segments [[DateSection]] and
    *         returns a stream of (t, Y(t)) where Y(t) is the signal/quantity chosen in
    *         [[OmniOSA.targetColumn]]
    * */
  def processTimeSegment = DataPipe((dateLimits: DateSection) => {
    //For the given date limits generate the relevant data
    //The file name to read from
    val fileName = dataDir+"omni2_"+dateLimits._1.split("/").head+".csv"

    val (startStamp, endStamp) = (
      formatter.parseDateTime(dateLimits._1).minusHours(0).getMillis/1000.0,
      formatter.parseDateTime(dateLimits._2).getMillis/1000.0)

    //Set up a processing pipeline for the file
    val processFile = omniFileToStream >
      extractTimeSeries((year,day,hour) => {
        dayofYearformatter.parseDateTime(
          year.toInt.toString + "/" + day.toInt.toString +
            "/" + hour.toInt.toString).getMillis/1000.0 }) >
      StreamDataPipe((couple: (Double, Double)) =>
        couple._1 >= startStamp && couple._1 <= endStamp)

    //Apply the pipeline to the file
    processFile(fileName)
  })

  val prepareData = DataPipe((s: Stream[(Double, Double)]) => {
    val tmin = s.map(_._1).min
    //subtract tmin from time index of each data point
    s.map(c => (c._1-tmin, c._2))
  })

  val getIncrements = DataPipe((s: Stream[(Double, Double)]) => {
    s.sliding(2).map(ss => (ss.last._1, ss.last._2 - ss.head._2))
  })


}
