import io.github.mandar2812.PlasmaML.omni.{DstARExperiment, DstARXExperiment, TestOmniARX}
import io.github.mandar2812.dynaml.kernels.{DiracKernel, PolynomialKernel}

/**
  * Created by mandar on 9/6/16.
  */


DstARExperiment("2008/01/01/00", "2008/01/11/10", new PolynomialKernel(2, 0.008), List(6),
  Map("globalOpt"->"GS","grid"->"2", "step"->"0.003", "fileID"->"Pred",
    "validationStart" -> "2014/11/15/00", "validationEnd" -> "2014/12/01/23",
    "action" -> "predict"))

DstARXExperiment("2008/01/01/00", "2008/01/11/00",
  new PolynomialKernel(2, 0.008), new DiracKernel(5.002),
  List(6,1,1,1), column = 40, ex = List(16,24),
  Map("globalOpt"->"GS","grid"->"2", "step"->"0.003", "fileID"->"Pred",
    "Use VBz" -> "true", "validationStart" -> "2014/11/15/00",
    "validationEnd" -> "2014/12/01/23", "action" -> "predict"))


TestOmniARX.runExperiment("2008/01/01/00", "2008/01/13/00", "2001/10/27/03", "2001/10/29/22",
  new PolynomialKernel(2,12.002), List(2,2,2,2), 1, new DiracKernel(5.002), 40, List(16, 24, 15),
  10, 0.2, "GS", Map("block" -> "degree", "Use VBz" -> "false"), "test")



DstARXExperiment("2008/01/01/00", "2008/01/11/00", new PolynomialKernel(2,12.002), new DiracKernel(5.002),
  List(2,2,2,2), column = 40, ex = List(16, 24, 15), Map("globalOpt"->"GS","grid"->"10", "step"->"0.003",
    "fileID"->"Poly_2", "action" -> "test", "block" -> "degree", "Use VBz" -> "false"))


TestOmniARX.runExperiment("2008/01/01/00", "2008/01/11/00", "2001/10/28/03", "2001/10/29/22",
  new PolynomialKernel(1, 10.002), List(6,1,1,1), 1, new DiracKernel(10.002), 40,
  List(16, 24), 10, 1.0, "GS", Map("Use VBz" -> "true", "validationStart" -> "2014/11/15/00",
    "validationEnd" -> "2014/12/01/23", "block" -> "degree"), "energyLandscape")
