import nom.tam.fits._
import _root_.io.github.mandar2812.PlasmaML.helios.fte
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.{utils => dutils}

//Some hard-coded meta data for FTE/Bss files.

//The latitude discretization
val latitude_grid = {
  dutils
    .range[Double](-1d, 1d, 360)
    .map(math.asin)
    .map(math.toDegrees)
    .zipWithIndex
    .filter(c => c._2 > 0 && c._2 <= 359 && c._2 % 2 == 1)
    .map(_._1)
    .map(x => BigDecimal.binary(x, new java.math.MathContext(4)).toDouble)
}

//The longitude discretization
val longitude_grid = (1 to 360).map(_.toDouble)

val files = dtfdata.dataset {
  List(
    //root / 'Users / 'mandar / 'Downloads / "GONGfte_csss2205HR.fits",
    root / 'Users / 'mandar / 'Downloads / "GONGbr1_csss2205HR.fits",
    root / 'Users / 'mandar / 'Downloads / "GONGbrcp_csss2205HR.fits",
    root / 'Users / 'mandar / 'Downloads / "GONGbrss_csss2205HR.fits"
  )
}

val file_to_array = DataPipe((file: Path) => {
  new Fits(file.toString)
    .getHDU(0)
    .getKernel()
    .asInstanceOf[Array[Array[Float]]]
    .toIterable
})

val array_to_patt = DataPipe(
  (arr: Iterable[Array[Float]]) =>
    latitude_grid
      .zip(arr)
      .flatMap(
        (s: (Double, Array[Float])) =>
          longitude_grid
            .zip(s._2)
            .map(
              (p: (Double, Float)) =>
                HelioPattern(p._1, s._1, Some(p._2.toDouble))
            )
      )
      .toIterable
)

files.flatMap(file_to_array > array_to_patt)
