import io.github.mandar2812.PlasmaML.omni.{DstARExperiment, DstARXExperiment, TestOmniAR, TestOmniARX}
import io.github.mandar2812.dynaml.kernels.{DiracKernel, PolynomialKernel}
import io.github.mandar2812.dynaml.pipes.DynaMLPipe

/**
  * @author mandar2812
  *
  * A script generating comparison results of
  * Dst OSA prediction on a list of 63 geomagnetic storms
  */


DstARExperiment("2008/01/01/00", "2008/01/11/10", new PolynomialKernel(2, 0.008), new DiracKernel(1.5),
  List(6), Map("globalOpt"->"GS","grid"->"2", "step"->"0.003", "fileID"->"Pred",
    "validationStart" -> "2014/11/15/00", "validationEnd" -> "2014/12/01/23",
    "action" -> "predict", "logScale" -> "false"))

DstARXExperiment("2008/01/01/00", "2008/01/11/00",
  new PolynomialKernel(1, 0.008), new DiracKernel(5.002),
  List(6,1,1,1), column = 40, ex = List(16,24),
  Map("globalOpt"->"GS","grid"->"2", "step"->"0.003", "fileID"->"Pred",
    "Use VBz" -> "true", "validationStart" -> "2014/11/15/00",
    "validationEnd" -> "2014/12/01/23", "action" -> "predict",
    "logScale" -> "false"))


TestOmniARX.runExperiment("2008/01/01/00", "2008/01/13/00", "2001/10/27/03", "2001/10/29/22",
  new PolynomialKernel(2,12.002), List(2,2,2,2), 1, new DiracKernel(5.002), 40, List(16, 24, 15),
  10, 0.2, "GS", Map("block" -> "degree", "Use VBz" -> "false", "logScale" -> "false"), "test")



DstARXExperiment("2008/01/01/00", "2008/01/11/00", new PolynomialKernel(2,12.002), new DiracKernel(5.002),
  List(2,2,2,2), column = 40, ex = List(16, 24, 15), Map("globalOpt"->"GS","grid"->"10", "step"->"0.003",
    "fileID"->"Poly_2", "action" -> "test", "block" -> "degree", "Use VBz" -> "false", "logScale" -> "false"))


TestOmniARX.runExperiment("2008/01/01/00", "2008/01/11/00", "2001/10/28/03", "2001/10/29/22",
  new PolynomialKernel(1, 0.1), List(6,1,1,1), 1, new DiracKernel(5.002), 40,
  List(16, 24), 10, 1.0, "GS", Map(
    "Use VBz" -> "true", "validationStart" -> "2014/11/15/00",
    "validationEnd" -> "2014/12/01/23", "block" -> "degree", "logScale" -> "true"),
  "energyLandscape")

//smaller grid and block hyper-parameter b

TestOmniARX.runExperiment("2008/01/01/00", "2008/01/11/00", "2001/10/28/03", "2001/10/29/22",
  new PolynomialKernel(1, 0.1), List(6,1,1,1), 1, new DiracKernel(5.002), 40,
  List(16, 24), 15, 1.0, "GS", Map(
    "Use VBz" -> "true", "validationStart" -> "2014/11/15/00",
    "validationEnd" -> "2014/12/01/23", "block" -> "degree", "logScale" -> "true"),
  "energyLandscape")

//Incorporating knowledge of landscape for new GP-ARX experiment

DstARXExperiment("2008/01/01/00", "2008/01/11/00",
  new PolynomialKernel(1, 0.001), new DiracKernel(0.55),
  List(6,1,1,1), column = 40, ex = List(16,24),
  Map("globalOpt"->"GS","grid"->"2", "step"->"0.2", "fileID"->"Exp_Set2_",
    "Use VBz" -> "true", "validationStart" -> "2014/11/15/00", "block" -> "degree",
    "validationEnd" -> "2014/12/01/23", "action" -> "test",
    "logScale" -> "true"))


// Plotting landscape for GP-AR
val pipeAR = DynaMLPipe.valuesToFile("data/LandscapeARRes.csv")

val landscapeAR = TestOmniAR.runExperiment("2008/01/01/00", "2008/01/11/00", "2001/10/28/03", "2001/10/29/22",
  new PolynomialKernel(1, 1.5), 6, 0, 0, new DiracKernel(3.5), 40,
  15, 0.1, "GS", Map(
    "validationStart" -> "2014/11/15/00", "validationEnd" -> "2014/12/01/23",
    "block" -> "degree", "logScale" -> "false"),
  "energyLandscape")


pipeAR(landscapeAR.toStream)

DstARExperiment(
  "2008/01/01/00", "2008/01/11/10", new PolynomialKernel(1, 0.12),
  new DiracKernel(3.0), List(6), Map("globalOpt"->"GS","grid"->"2", "step"->"0.2",
    "fileID"->"Exp_Set2_","validationStart" -> "2014/11/15/00", "validationEnd" -> "2014/12/01/23",
      "action" -> "test", "logScale" -> "true", "block" -> "degree"))
