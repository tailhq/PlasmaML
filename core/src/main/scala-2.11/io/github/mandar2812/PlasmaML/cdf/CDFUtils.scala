package io.github.mandar2812.PlasmaML.cdf

import java.io.File

import io.github.mandar2812.PlasmaML.PlasmaML
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.spark.rdd.RDD

/**
  * Created by mandar on 20/5/16.
  */
object CDFUtils {

  val dataDir = "data/"

  var leapSecondsFile: String = dataDir+"CDFLeapSeconds.txt"

  var cdfBufferSize: Int = 100000

  var epochFormat: String = "TT2000"

  def epochProcessor = epochFormat match {
    case "TT2000" => new EpochFormatter().formatTimeTt2000 _
    case _ => new EpochFormatter().formatTimeTt2000 _
  }

  def getVariableAttributes(cdf: CdfContent) =
    cdf.getVariables.map(v => (
      v.getName,
      cdf.getVariableAttributes.map(attr =>
        (attr.getName, attr.getEntry(v) match {
            case null => ""
            case obj => obj.toString
          })
      ).toMap)
    ).toMap

  def cdfAttributes(file: String) = getVariableAttributes(new CdfContent(new CdfReader(new File(file))))

  def readCDF = DataPipe((file: String) => {
    val content = new CdfContent(new CdfReader(new File(file)))
    (content, getVariableAttributes(content))
  })

  def cdfToRDD(columns: Seq[String], missingValueKey: String = "FILLVAL") = DataPipe(
    (cdfContentAndAttributes: (CdfContent, Map[String, Map[String, String]])) => {
      val (content, columnData) = cdfContentAndAttributes
      val variablesSelected = content.getVariables
        .filter(v => columns.contains(v.getName))
        .sortBy(v => columns.indexOf(v.getName))

      variablesSelected.map(v => {
        val rawValueArray = v.createRawValueArray()
        (0 until v.getRecordCount).grouped(cdfBufferSize)
          .map(buffer => PlasmaML.sc.parallelize(
            buffer.map(index => Seq(v.readShapedRecord(index, false, rawValueArray).toString))
          )).reduceLeft((a, b) => a union b)
      }).reduceLeft((rddL, rddR) =>
        (rddL zip rddR) mapPartitions (_.map(couple => couple._1 ++ couple._2)))
    })

  def processEpochs(position: Int = 0) =
    DataPipe((cdfRDD: RDD[Seq[String]]) => {
      cdfRDD.mapPartitions(_.map(
        _.zipWithIndex.map(couple =>
          if (couple._2 == position) epochProcessor(couple._1.toLong)
          else couple._1))
      )
    })
}