import io.github.mandar2812.PlasmaML.omni._
import io.github.mandar2812.dynaml.kernels.{DiracKernel, PolynomialKernel}
import io.github.mandar2812.dynaml.pipes.DynaMLPipe

/**
  * @author mandar2812
  *
  * A script generating comparison results of
  * Dst OSA prediction on a list of 63 geomagnetic storms
  */


val (trainingStart, trainingEnd) = ("2008/01/01/00", "2008/01/11/10")

val (validationStart, validationEnd) = ("2011/08/05/15", "2011/08/06/22")

//Generate hourly predictions for all storms in Ji et al 2012
DstARExperiment(trainingStart, trainingEnd,
  new PolynomialKernel(1, 0.0), new DiracKernel(1.75),
  List(6), Map("globalOpt"->"GS","grid"->"2", "step"->"0.2", "fileID"->"Pred",
    "validationStart" -> "2014/11/15/00", "validationEnd" -> "2014/12/01/23",
    "action" -> "predict", "logScale" -> "true"))

DstARXExperiment(trainingStart, trainingEnd,
  new PolynomialKernel(1, 0.0), new DiracKernel(0.55),
  List(6,1,1,1), column = 40, ex = List(16,24),
  Map("globalOpt"->"GS","grid"->"2", "step"->"0.003", "fileID"->"Pred",
    "Use VBz" -> "true", "validationStart" -> "2014/11/15/00",
    "validationEnd" -> "2014/12/01/23", "action" -> "predict",
    "logScale" -> "true"))

TestOmniTL.prepareTLFiles()

DstNMTLExperiment("TL", "predict", "Pred")

DstNMTLExperiment("NM", "predict", "Pred")

DstPersistExperiment("predict", "Pred")

// Generate predictions for sample storm in November 2004
val (start2004Storm, end2004Storm) = ("2004/11/09/11", "2004/11/11/09")

val pipeARXErr = DynaMLPipe.valuesToFile("data/ARXErrorBarsFinalPredRes.csv")
val pred2004ARXErr = TestOmniARX.runExperiment(trainingStart, trainingEnd, start2004Storm, end2004Storm,
  new PolynomialKernel(1, 0.0), List(6,1,1,1), 1, new DiracKernel(0.55), 40,
  List(16, 24), 2, 0.2, "GS", Map(
    "Use VBz" -> "true", "validationStart" -> validationStart,
    "validationEnd" -> validationEnd, "block" -> "degree,offset",
    "logScale" -> "true"),
  "predict_error_bars")

pipeARXErr(pred2004ARXErr.toStream)

val pipeARErr = DynaMLPipe.valuesToFile("data/ARErrorBarsFinalPredRes.csv")
val pred2004ARErr = TestOmniAR.runExperiment(trainingStart, trainingEnd, start2004Storm, end2004Storm,
  new PolynomialKernel(1, 0.0), 6, 0, 0, new DiracKernel(1.75), 40,
  2, 0.2, "GS", Map(
    "validationStart" -> validationStart, "validationEnd" -> validationEnd,
    "block" -> "degree,offset", "logScale" -> "true"),
  "predict_error_bars")
pipeARErr(pred2004ARErr.toStream)


val pipeNM = DynaMLPipe.valuesToFile("data/NMFinalPredRes.csv")
val nmPred = TestOmniNarmax(start2004Storm, end2004Storm, "predict")
pipeNM(nmPred)

val pipeTL = DynaMLPipe.valuesToFile("data/TLFinalPredRes.csv")
val tlPred = TestOmniTL(start2004Storm, end2004Storm, "predict")
pipeTL(tlPred)
// Generate results for Persist(1)

DstPersistExperiment("test")

// Experiment ARX Set 3

DstARXExperiment(trainingStart, trainingEnd,
  new PolynomialKernel(1, 0.0), new DiracKernel(0.55),
  List(6,1,1,1), column = 40, ex = List(16,24),
  Map("globalOpt"->"GS","grid"->"2", "step"->"0.2", "fileID"->"Exp_Set3_",
    "Use VBz" -> "true", "validationStart" -> validationStart, "block" -> "degree,offset",
    "validationEnd" -> validationEnd, "action" -> "test",
    "logScale" -> "true"))

//Experiment AR Set 3

DstARExperiment(trainingStart, trainingEnd,
  new PolynomialKernel(1, 0.0), new DiracKernel(1.75),
  List(6), Map("globalOpt"->"GS","grid"->"2", "step"->"0.2",
    "fileID"->"Exp_Set3_","validationStart" -> validationStart, "validationEnd" -> validationEnd,
    "action" -> "test", "logScale" -> "true", "block" -> "degree,offset"))

val orders = for(i <- 7 to 9; j <- 6 to 8) yield (i,j)


orders.foreach({
  case (yOrder, xOrder) =>
    val fileID = "Nov_4_2016_"+yOrder.toString()+"_"+xOrder.toString()+"_"
    DstARXExperiment(trainingStart, trainingEnd,
      new PolynomialKernel(1, 0.0), new DiracKernel(1.55),
      List(yOrder,xOrder,xOrder,xOrder), column = 40, ex = List(16,24),
      Map("globalOpt"->"GS","grid"->"2", "step"->"0.2", "fileID"-> fileID,
        "Use VBz" -> "true", "validationStart" -> validationStart, "block" -> "degree,offset",
        "validationEnd" -> validationEnd, "action" -> "test",
        "logScale" -> "false"))

})


val orders = for(i <- 2 to 7; j <- 1 to 6) yield (i,j)
orders.foreach({
  case (yOrder, xOrder) =>
    val fileID = "Nov_4_2016_alt_"+yOrder.toString()+"_"+xOrder.toString()+"_"
    DstARXExperiment.alternate_experiment(trainingStart, trainingEnd,
      new PolynomialKernel(1, 0.0), new DiracKernel(1.55),
      List(yOrder,xOrder,xOrder,xOrder), column = 40, ex = List(16,24),
      Map("globalOpt"->"GS","grid"->"2", "step"->"0.2", "fileID"-> fileID,
        "Use VBz" -> "true", "validationStart" -> validationStart, "block" -> "degree,offset",
        "validationEnd" -> validationEnd, "action" -> "test",
        "logScale" -> "false"))

})

