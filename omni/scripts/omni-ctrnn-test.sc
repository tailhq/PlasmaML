import _root_.io.github.mandar2812.dynaml.repl.Router.main
import org.joda.time._
import io.github.mandar2812.PlasmaML.omni.OMNIData.Quantities._
import io.github.mandar2812.PlasmaML.omni.{OMNIData, OMNILoader}
import io.github.mandar2812.dynaml.pipes.StreamDataPipe
import io.github.mandar2812.dynaml.tensorflow._

@main
def main(yearrange: Range, quantities: Seq[Int] = Seq(Dst, V_SW, B_Z), horizon: Int = 24) = {

  val process_omni_files = OMNILoader.omniDataToSlidingTS(0, horizon+1)(quantities.head, quantities.tail)

  val extract_features_and_targets = StreamDataPipe((d: (DateTime, Seq[Seq[Double]])) => {

    (d._1, d._2.map(s => (s.head, s.tail)).unzip)
  })

  val data_process_pipe = process_omni_files > extract_features_and_targets

  val training_data = data_process_pipe(yearrange.map(OMNIData.getFilePattern).map("data/"+_).toStream)

  val (features, labels) = training_data.unzip._2.unzip

  val n_data = features.length

  val (features_tf, labels_tf) = (
    dtf.tensor_f32(n_data, quantities.length)(features.flatten:_*),
    dtf.tensor_f32(n_data, quantities.length, horizon)(labels.flatMap(_.flatten):_*)
  )

  dtfpipe.gaussian_standardization(features_tf, labels_tf)

}

def apply(yearrange: Range, quantities: Seq[Int] = Seq(Dst, V_SW, B_Z), horizon: Int = 24) =
  main(yearrange, quantities, horizon)