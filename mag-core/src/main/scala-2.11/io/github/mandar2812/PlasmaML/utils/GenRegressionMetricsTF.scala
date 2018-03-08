package io.github.mandar2812.PlasmaML.utils

import com.quantifind.charts.Highcharts.{regression, title, xAxis, yAxis}
import io.github.mandar2812.dynaml.evaluation.RegressionMetricsTF
import io.github.mandar2812.dynaml.tensorflow.dtf
import org.platanios.tensorflow.api._

//TODO: Modify [[RegressionMetricsTF]] to incorporate the idea here
class GenRegressionMetricsTF(preds: Tensor, targets: Tensor) extends RegressionMetricsTF(preds, targets) {
  private val num_outputs =
    if (preds.shape.toTensor().size == 1) 1
    else preds.shape.toTensor()(0 :: -1).prod().scalar.asInstanceOf[Int]

  private lazy val (_ , rmse , mae, corr) = GenRegressionMetricsTF.calculate(preds, targets)

  private lazy val modelyield =
    (preds.max(axes = 0) - preds.min(axes = 0)).divide(targets.max(axes = 0) - targets.min(axes = 0))

  override protected def run(): Tensor = dtf.stack(Seq(rmse, mae, corr, modelyield), axis = -1)

  override def generatePlots(): Unit = {
    println("Generating Plot of Fit for each target")

    if(num_outputs == 1) {
      val (pr, tar) = (
        scoresAndLabels._1.entriesIterator.map(_.asInstanceOf[Float]),
        scoresAndLabels._2.entriesIterator.map(_.asInstanceOf[Float]))

      regression(pr.zip(tar).toSeq)

      title("Goodness of fit: "+name)
      xAxis("Predicted "+name)
      yAxis("Actual "+name)

    } else {
      (0 until num_outputs).foreach(output => {
        val (pr, tar) = (
          scoresAndLabels._1(::, output).entriesIterator.map(_.asInstanceOf[Float]),
          scoresAndLabels._2(::, output).entriesIterator.map(_.asInstanceOf[Float]))

        regression(pr.zip(tar).toSeq)
      })
    }
  }
}

object GenRegressionMetricsTF {

  protected def calculate(preds: Tensor, targets: Tensor): (Tensor, Tensor, Tensor, Tensor) = {
    val error = targets.subtract(preds)

    println("Shape of error tensor: "+error.shape.toString()+"\n")

    val num_instances = error.shape(0)
    val rmse = error.square.mean(axes = 0).sqrt

    val mae = error.abs.mean(axes = 0)

    val corr = {

      val mean_preds = preds.mean(axes = 0)

      val mean_targets = targets.mean(axes = 0)

      val preds_c = preds.subtract(mean_preds)

      val targets_c = targets.subtract(mean_targets)

      val (sigma_t, sigma_p) = (targets_c.square.mean(axes = 0).sqrt, preds_c.square.mean(axes = 0).sqrt)

      preds_c.multiply(targets_c).mean(axes = 0).divide(sigma_t.multiply(sigma_p))
    }

    (error, rmse, mae, corr)
  }
}