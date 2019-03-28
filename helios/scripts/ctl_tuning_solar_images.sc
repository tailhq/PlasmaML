import ammonite.ops._
import org.joda.time._
import io.github.mandar2812.dynaml.repl.Router.main
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, Encoder}
import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}
import _root_.io.github.mandar2812.PlasmaML.helios
import _root_.io.github.mandar2812.PlasmaML.helios.data
import io.github.mandar2812.PlasmaML.helios.data.{SDO, SOHO, SOHOData, SolarImagesSource}
import io.github.mandar2812.PlasmaML.helios.data.SDOData.Instruments._
import io.github.mandar2812.PlasmaML.helios.data.SOHOData.Instruments._
import _root_.io.github.mandar2812.dynaml.tensorflow.layers.{L1Regularization, L2Regularization}
import breeze.numerics.sigmoid
import io.github.mandar2812.PlasmaML.helios.core.timelag
import io.github.mandar2812.dynaml.probability.UniformRV
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.NN.SameConvPadding


@main
def apply[T <: SolarImagesSource](
  year_range: Range                                     = 2001 to 2004,
  test_year: Int                                        = 2003,
  ff_stack_sizes: Seq[Int]                              = Seq(256, 128, 64),
  image_source: T                                       = SOHO(MDIMAG, 512),
  buffer_size: Int                                      = 2000,
  re: Boolean                                           = true,
  scaleDown: Int                                        = 4,
  time_horizon: (Int, Int)                              = (24, 100),
  image_hist: Int                                       = 0,
  image_hist_downsamp: Int                              = 1,
  opt: Optimizer                                        = tf.train.AdaDelta(0.01f),
  iterations: Int                                       = 1000000,
  iterations_tuning: Int                                = 50000,
  divergence: helios.learn.cdt_loss.Divergence          = helios.learn.cdt_loss.KullbackLeibler,
  target_dist: helios.learn.cdt_loss.TargetDistribution = helios.learn.cdt_loss.Boltzmann,
  miniBatch: Int                                        = 16,
  tmpdir: Path                                          = root/"home"/System.getProperty("user.name")/"tmp",
  path_to_images: Option[Path]                          = None,
  regularization_type: String                           = "L2",
  hyper_optimizer: String                               = "gs",
  num_hyp_samples: Int                                  = 20,
  hyp_opt_iterations: Option[Int]                       = Some(5),
  existing_exp: Option[Path]                            = None)
: helios.Experiment[Double, helios.ModelRunTuning[Double, Double], helios.ImageExpConfig] = {

  //Data with MDI images

  print("Running experiment with test split from year: ")
  pprint.pprintln(test_year)

  data.buffer_size_(buffer_size)

  val dataset = data.generate_data_omni[T](
    year_range, image_source,
    deltaT = time_horizon,
    images_data_dir = path_to_images)

  println("Starting data set created.")
  println("Proceeding to load images & labels into Tensors ...")
  val sw_threshold   = 0.0

  val test_start     = new DateTime(test_year, 1, 1, 0, 0)
  val test_end       = new DateTime(test_year, 12, 31, 23, 59)

  val tt_partition   = (p: (DateTime, (Seq[Path], Seq[Double]))) =>
    if (p._1.isAfter(test_start) && p._1.isBefore(test_end) && p._2._2.max >= sw_threshold) false
    else true


  val (image_sizes, magic_ratio) = image_source match {
    case SOHO(_, s) => (s, 268.0/512.0)
    case SDO(_, s)  => (s, 333.0/512.0)
  }

  val (image_filter, num_channels, image_to_byte) = data.image_process_metadata(image_source)

  val patch_range = data.get_patch_range(/*magic_ratio*/1.0, image_sizes/scaleDown)

  val image_preprocess = /*data.image_central_patch(magic_ratio, image_sizes) >*/ data.image_scale(1.0/scaleDown)

  //Set the path of the summary directory
  val summary_dir_prefix  = "swtl_"+image_source.toString
  val dt                  = DateTime.now()
  val summary_dir_postfix =
    if(re) "_re_"+dt.toString("YYYY-MM-dd-HH-mm")
    else "_"+dt.toString("YYYY-MM-dd-HH-mm")

  //val (summary_dir , reuse): (String, Boolean)  =
  //  if(existingModelDir.isEmpty) (summary_dir_prefix+summary_dir_postfix, false)
  //  else (existingModelDir, true)

  //if(reuse) println("\nReusing existing model in directory: "+existingModelDir+"\n")

  //Set some meta-information for the prediction architecture

  val summary_dir      = summary_dir_prefix+summary_dir_postfix

  val num_pred_dims    = 2*dataset.data.head._2._2.length
  val ff_index         = 6
  val ff_stack         = ff_stack_sizes :+ num_pred_dims

  /*
  * Construct architecture components.
  *
  * 1) Convolutional segment: Consists of two convolutional pyramids interspersed by batch normalisation
  * 2) Pre Upwind feed-forward segment.
  * 3) Upwind 1D computational layer.
  * 4) Post Upwind feed-forward segment.
  * 5) A post processing output mapping.
  * */

  val filter_depths_stack1 = Seq(
    Seq(15, 15, 15, 15),
    Seq(10, 10, 10, 10)
  )

  val filter_depths_stack2 = Seq(
    Seq(5, 5, 5, 5),
    Seq(1, 1, 1, 1)
  )

  val activation = DataPipe[String, Layer[Output[Float], Output[Float]]]((s: String) => tf.learn.ReLU(s, 0.01f))

  val conv_section =
    dtflearn.inception_stack(
      num_channels*(image_hist_downsamp + 1),
      filter_depths_stack1, activation,
      use_batch_norm = true)(1) >>
    tf.learn.MaxPool(s"MaxPool_1", Seq(1, 3, 3, 1), 2, 2, SameConvPadding) >>
    dtflearn.inception_stack(
      filter_depths_stack1.last.sum, filter_depths_stack2,
      activation, use_batch_norm = true)(5) >>
    tf.learn.MaxPool(s"MaxPool_2", Seq(1, 3, 3, 1), 2, 2, SameConvPadding)


  val post_conv_ff_stack = dtflearn.feedforward_stack[Double](
    (i: Int) => tf.learn.Sigmoid("Act_"+i))(
    ff_stack, starting_index = ff_index)

  val output_mapping = helios.learn.cdt_loss.output_mapping[Double](
    name = "Output/CDT-SW",
    dataset.data.head._2._2.length)


  val architecture =
    tf.learn.Cast[UByte, Float]("Cast/Input") >>
      conv_section >>
      tf.learn.Flatten("Flatten_1") >>
      tf.learn.Cast[Float, Double]("Cast/Input") >>
      post_conv_ff_stack >>
      output_mapping


  val (_, layer_shapes, layer_parameter_names, layer_datatypes) =
    dtfutils.get_ffstack_properties(
      -1,
      num_pred_dims,
      ff_stack.dropRight(1),
      dType = "FLOAT64",
      starting_index = ff_index)

  val scope = dtfutils.get_scope(architecture) _

  val layer_scopes = layer_parameter_names.map(n => scope(n.split("/").head))

  val hyper_parameters = List(
    "prior_wt",
    "error_wt",
    "temperature",
    "specificity",
    "reg"
  )

  val hyper_prior = Map(
    "prior_wt"    -> UniformRV(0.5, 1.5),
    "error_wt"    -> UniformRV(0.75, 1.5),
    "temperature" -> UniformRV(1.75, 2.5),
    "specificity" -> UniformRV(0.5, 2.5),
    "reg"         -> UniformRV(math.pow(10d, -5d), math.pow(10d, -3d))
  )


  //The following definitions are needed in case of CMA-ES and CSA methods
  val hyp_scaling = hyper_prior.map(p =>
    (
      p._1,
      Encoder((x: Double) => (x - p._2.min)/(p._2.max - p._2.min), (u: Double) => u*(p._2.max - p._2.min) + p._2.min)
    )
  )

  val logit = Encoder((x: Double) => math.log(x/(1d - x)), (x: Double) => sigmoid(x))

  val hyp_mapping = Some(
    hyper_parameters.map(
      h => (h, hyp_scaling(h) > logit)
    ).toMap
  )



  //Generate the loss function given values for the hyper-parameters
  val loss_func_generator = (h: Map[String, Double]) => {

    val lossFunc = timelag.utils.get_loss[Double, Double, Double](
      time_horizon._2, mo_flag = true,
      prob_timelags = true,
      prior_wt = h("prior_wt"),
      prior_divergence = divergence,
      target_dist = target_dist,
      temp = h("temperature"),
      error_wt = h("error_wt"),
      c = h("specificity"))

    val reg_layer =
      if(regularization_type == "L1")
        L1Regularization[Double](layer_scopes, layer_parameter_names, layer_datatypes, layer_shapes, h("reg"))
      else
        L2Regularization[Double](layer_scopes, layer_parameter_names, layer_datatypes, layer_shapes, h("reg"))

    lossFunc >>
      reg_layer >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")
  }

  val fitness_function = DataPipe2[(Output[Double], Output[Double]), Output[Double], Output[Float]]((preds, targets) => {

    val weighted_error = preds._1
      .subtract(targets)
      .square
      .multiply(preds._2)
      .sum(axes = 1)

    val entropy = preds._2
      .multiply(Tensor(-1d).castTo[Double])
      .multiply(tf.log(preds._2))
      .sum(axes = 1)

      (weighted_error + entropy).castTo[Float]
  })

  val experiment = helios.run_cdt_experiment_omni_hyp(
    dataset, tt_partition, resample = re,
    preprocess_image = image_preprocess > image_filter,
    image_to_bytearr = image_to_byte,
    num_channels_image = num_channels,
    image_history = image_hist,
    image_history_downsampling = image_hist_downsamp,
    processed_image_size = (patch_range.length - 1, patch_range.length - 1),
    summaries_top_dir = tmpdir, results_id = summary_dir,
    architecture = architecture,
    loss_func_generator = loss_func_generator,
    hyper_params = hyper_parameters,
    fitness_func = fitness_function,
    hyper_prior = hyper_prior,
    optimizer = opt,
    iterations = iterations,
    iterations_tuning = iterations_tuning,
    hyper_optimizer = hyper_optimizer,
    hyp_mapping = hyp_mapping,
    hyp_opt_iterations = hyp_opt_iterations,
    num_hyp_samples = num_hyp_samples, 
    existing_exp = existing_exp)

  experiment.copy(config = experiment.config.copy(image_sources = Seq(image_source)))
}
