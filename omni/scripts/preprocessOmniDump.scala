import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes.StreamDataPipe
import org.joda.time.DateTimeZone
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}



DateTimeZone.setDefault(DateTimeZone.UTC)
val formatter: DateTimeFormatter = DateTimeFormat.forPattern("yyyy/MM/dd/HH")
val dayofYearformat = DateTimeFormat.forPattern("yyyy/D/H")

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

dataDump(Stream((("2008/01/01/00", "2008/01/11/10"), "osa_training.csv"), (("2014/11/15/00", "2014/12/01/23"), "osa_validation.csv")))
