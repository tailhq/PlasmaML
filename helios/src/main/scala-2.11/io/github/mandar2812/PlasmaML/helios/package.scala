package io.github.mandar2812.PlasmaML

import ammonite.ops.{Path, exists, home, ls, root, pwd}
import breeze.linalg.DenseVector
import org.joda.time._
import com.sksamuel.scrimage.Image

import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.{DiscreteDistrRV, MultinomialRV}
import io.github.mandar2812.dynaml.tensorflow.dtf

import _root_.io.github.mandar2812.PlasmaML.omni.{OMNILoader, OMNIData}
import _root_.io.github.mandar2812.PlasmaML.helios.core._
import _root_.io.github.mandar2812.PlasmaML.helios.data._

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.{Layer, Loss}

/**
  * <h3>Helios</h3>
  *
  * The top level package for the helios module.
  *
  * Contains methods for carrying out ML experiments
  * on data sets associated with helios.
  *
  * @author mandar2812
  * */
package object helios {


  object learn {

    val cnn_goes_v1: Layer[Output, Output]                 = Arch.cnn_goes_v1

    val cnn_goes_v1_1: Layer[Output, Output]               = Arch.cnn_goes_v1_1

    val cnn_sw_v1: Layer[Output, Output]                   = Arch.cnn_sw_v1

    /*
    * Loss Functions
    * */
    val weightedL2FluxLoss: (String) => WeightedL2FluxLoss =
      (name: String) => new WeightedL2FluxLoss(name)

    val rBFWeightedSWLoss: (String, Int) => RBFWeightedSWLoss =
      (name: String, horizon: Int) => new RBFWeightedSWLoss(name, horizon)

    val dynamicRBFSWLoss: (String, Int) => DynamicRBFSWLoss =
      (name: String, horizon: Int) => new DynamicRBFSWLoss(name, horizon)

  }

  /**
    * Download solar images from a specified source.
    *
    * @param source An instance of [[data.Source]], can be constructed
    *               using any of its subclasses,
    *               ex: [[data.SOHO]] or [[data.SDO]]
    *
    * @param download_path The location on disk where the data
    *                      is to be dumped.
    *
    * @param start The starting date from which images are extracted.
    *
    * @param end The date up to which images are extracted.
    * */
  def download_image_data(
    source: data.Source, download_path: Path)(
    start: LocalDate, end: LocalDate): Unit = source match {

    case data.SOHO(instrument, size) =>
      SOHOLoader.bulk_download(download_path)(instrument, size)(start, end)

    case data.SDO(instrument, size) =>
      SDOLoader.bulk_download(download_path)(instrument, size)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }

  /**
    * Download solar flux from the GOES data repository.
    *
    * @param source An instance of [[data.Source]], can be constructed
    *               using any of its subclasses,
    *               ex: [[data.GOES]]
    *
    * @param download_path The location on disk where the data
    *                      is to be dumped.
    *
    * @param start The starting year-month from which images are extracted.
    *
    * @param end The year-month up to which images are extracted.
    *
    * @throws Exception if the data source is not valid
    *                   (i.e. not [[data.GOES]])
    * */
  def download_flux_data(
    source: data.Source, download_path: Path)(
    start: YearMonth, end: YearMonth): Unit = source match {

    case data.GOES(quantity, format) =>
      GOESLoader.bulk_download(download_path)(quantity, format)(start, end)

    case _ =>
      throw new Exception("Not a valid data source: ")
  }

  def load_images(
    soho_files_path: Path, year_month: YearMonth,
    soho_source: SOHO, dirTreeCreated: Boolean = true) =
    SOHOLoader.load_images(soho_files_path, year_month, soho_source, dirTreeCreated)

  /**
    * Load X-Ray fluxes averaged over all GOES missions
    *
    * */
  def load_fluxes(
    goes_files_path: Path, year_month: YearMonth,
    goes_source: GOES, dirTreeCreated: Boolean = true) =
    GOESLoader.load_goes_data(
      goes_files_path, year_month,
      goes_source, dirTreeCreated)
      .map(p => {

        val data_low_wavelength = p._2.map(_._1).filterNot(_.isNaN)
        val data_high_wavelength = p._2.map(_._2).filterNot(_.isNaN)

        val avg_low_freq = data_low_wavelength.sum/data_low_wavelength.length
        val avg_high_freq = data_high_wavelength.sum/data_high_wavelength.length

        (p._1, (avg_low_freq, avg_high_freq))
    })

  /**
    * Collate data from GOES with Image data.
    *
    * @param goes_data_path GOES data path.
    *
    * @param images_path path containing images.
    *
    * @param goes_aggregation The number of goes entries to group for
    *                         calculating running statistics.
    *
    * @param goes_reduce_func A function which computes some aggregation of a group
    *                         of GOES data entries.
    *
    * @param dt_round_off A function which appropriately rounds off date time instances
    *                     for the image data, enabling it to be joined to the GOES data
    *                     based on date time stamps.
    * */
  def collate_goes_data(
    year_month: YearMonth)(
    goes_source: GOES,
    goes_data_path: Path,
    goes_aggregation: Int,
    goes_reduce_func: (Stream[(DateTime, (Double, Double))]) => (DateTime, (Double, Double)),
    image_source: SOHO, images_path: Path,
    dt_round_off: (DateTime) => DateTime, dirTreeCreated: Boolean = true) = {

    val proc_goes_data = load_fluxes(
      goes_data_path, year_month,
      goes_source, dirTreeCreated)
      .grouped(goes_aggregation).map(goes_reduce_func).toMap

    val proc_image_data = load_images(
      images_path, year_month,
      image_source, dirTreeCreated)
      .map(p => (dt_round_off(p._1), p._2)).toMap

    proc_image_data.map(kv => {

      val value =
        if (proc_goes_data.contains(kv._1)) Some(proc_goes_data(kv._1))
        else None

      (kv._1, (kv._2, value))})
      .toStream
      .filter(k => k._2._2.isDefined)
      .map(k => (k._1, (k._2._1, k._2._2.get)))
      .sortBy(_._1.getMillis)
  }

  /**
    * Calls [[collate_goes_data()]] over a time period and returns the collected data.
    *
    * @param start_year_month Starting Year-Month
    * @param end_year_month Ending Year-Month
    * @param goes_data_path GOES data path.
    * @param images_path path containing images.
    * @param goes_aggregation The number of goes entries to group for
    *                         calculating running statistics.
    * @param goes_reduce_func A function which computes some aggregation of a group
    *                         of GOES data entries.
    * @param dt_round_off A function which appropriately rounds off date time instances
    *                     for the image data, enabling it to be joined to the GOES data
    *                     based on date time stamps.
    * */
  def collate_goes_data_range(
    start_year_month: YearMonth, end_year_month: YearMonth)(
    goes_source: GOES,
    goes_data_path: Path,
    goes_aggregation: Int,
    goes_reduce_func: (Stream[(DateTime, (Double, Double))]) => (DateTime, (Double, Double)),
    image_source: SOHO, images_path: Path,
    dt_round_off: (DateTime) => DateTime,
    dirTreeCreated: Boolean = true) = {

    val prepare_data = (ym: YearMonth) => collate_goes_data(ym)(
      goes_source, goes_data_path, goes_aggregation, goes_reduce_func,
      image_source, images_path, dt_round_off)

    val period = new Period(
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay)

    print("Time period considered (in months): ")

    val num_months = (12*period.getYears) + period.getMonths

    pprint.pprintln(num_months)

    (0 to num_months).map(start_year_month.plusMonths).flatMap(prepare_data).toStream
  }

  /**
    *
    *
    * */
  def collate_omni_data_range(
    start_year_month: YearMonth,
    end_year_month: YearMonth)(
    omni_source: OMNI, omni_data_path: Path, deltaT: (Int, Int),
    image_source: SOHO, images_path: Path,
    image_dir_tree: Boolean = true) = {

    val (start_instant, end_instant) = (
      start_year_month.toLocalDate(1).toDateTimeAtStartOfDay,
      end_year_month.toLocalDate(31).toDateTimeAtStartOfDay
    )

    val period = new Period(start_instant, end_instant)


    print("Time period considered (in months): ")

    val num_months = (12*period.getYears) + period.getMonths

    pprint.pprintln(num_months)


    //Extract OMNI data as stream

    //First create the transformation pipe

    val omni_processing =
      OMNILoader.omniDataToSlidingTS(deltaT._1, deltaT._2)(OMNIData.Quantities.V_SW) >
        StreamDataPipe[(DateTime, Seq[Double])](
          (p: (DateTime, Seq[Double])) => p._1.isAfter(start_instant) && p._1.isBefore(end_instant)
        )

    val years = (start_year_month.getYear to end_year_month.getYear).toStream

    val omni_data = omni_processing(years.map(i => omni_data_path.toString()+"/omni2_"+i+".csv"))

    //Extract paths to images, along with a time-stamp

    val image_dt_roundoff: (DateTime) => DateTime = (d: DateTime) => {
      new DateTime(d.getYear, d.getMonthOfYear, d.getDayOfMonth, d.getHourOfDay)
    }

    val image_processing =
      StreamFlatMapPipe(
        (year_month: YearMonth) => load_images(images_path, year_month, image_source, image_dir_tree)) >
      StreamDataPipe((p: (DateTime, Path)) => (image_dt_roundoff(p._1), p._2))

    val images = image_processing((0 to num_months).map(start_year_month.plusMonths).toStream).toMap

    omni_data.map(o => {
      val image_option = images.get(o._1)
      (o._1, image_option, o._2)
    }).filter(_._2.isDefined)
      .map(d => (d._1, (d._2.get, d._3)))

  }

  /**
    * Resample data according to a provided
    * bounded discrete random variable
    * */
  def resample[T](
    data: Stream[(DateTime, (Path, T))],
    selector: DiscreteDistrRV[Int]): Stream[(DateTime, (Path, T))] = {

    //Resample training set ot
    //emphasize extreme events.
    println("\nResampling data instances\n")

    selector.iid(data.length).draw.map(data(_))
  }

  /**
    * Create a processed tensor data set as a [[HeliosDataSet]] instance.
    *
    * @param collated_data A Stream of date times, image paths and fluxes.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * @param scaleDownFactor The exponent of 2 which determines how much the
    *                        image will be scaled down. i.e. scaleDownFactor = 4
    *                        corresponds to a 16 fold decrease in image size.
    * */
  def create_helios_data_set(
    collated_data: Stream[(DateTime, (Path, Seq[Double]))],
    tt_partition: ((DateTime, (Path, Seq[Double]))) => Boolean,
    scaleDownFactor: Int = 4, resample: Boolean = false): HeliosDataSet = {

    val scaleDown = 1/math.pow(2, scaleDownFactor)

    print("Scaling down images by a factor of ")
    pprint.pprintln(math.pow(2, scaleDownFactor))
    println()

    println("Separating data into train and test.")
    val (train_set, test_set) = collated_data.partition(tt_partition)

    //Calculate the height, width and number of channels
    //in the images
    val (scaled_height, scaled_width, num_channels) = {

      val im = Image.fromPath(train_set.head._2._1.toNIO)

      val scaled_image = im.copy.scale(scaleDown)

      (scaled_image.height, scaled_image.width, scaled_image.argb(0, 0).length)

    }

    val working_set = HeliosDataSet(
      null, null, train_set.length,
      null, null, test_set.length)

    /*
    * If the `resample` flag is set to true,
    * balance the occurence of high and low
    * flux events through re-sampling.
    *
    * */
    val processed_train_set = if(resample) {
      //Resample training set ot
      //emphasize extreme events.

      val un_prob = train_set.map(_._2._2.sum).map(math.exp)
      val normalizer = un_prob.sum
      val selector = MultinomialRV(DenseVector(un_prob.toArray)/normalizer)

      helios.resample(train_set, selector)
    } else train_set

    val (features_train, labels_train): (Stream[Array[Byte]], Stream[Seq[Double]]) =
      processed_train_set.map(entry => {
        val (_, (path, data_label)) = entry

        val im = Image.fromPath(path.toNIO)

        val scaled_image = im.copy.scale(scaleDown)

        (scaled_image.argb.flatten.map(_.toByte), data_label)

      }).unzip

    val features_tensor_train = dtf.tensor_from_buffer(
      "UINT8", processed_train_set.length, scaled_height, scaled_width, num_channels)(
      features_train.toArray.flatten[Byte])

    val labels_tensor_train = dtf.tensor_from("FLOAT32", train_set.length, 2)(labels_train.flatten[Double])


    val (features_test, labels_test): (Stream[Array[Byte]], Stream[Seq[Double]]) = test_set.map(entry => {
      val (_, (path, data_label)) = entry

      val im = Image.fromPath(path.toNIO)

      val scaled_image = im.copy.scale(scaleDown)

      (scaled_image.argb.flatten.map(_.toByte), data_label)

    }).unzip

    val features_tensor_test = dtf.tensor_from_buffer(
      "UINT8", test_set.length, scaled_height, scaled_width, num_channels)(
      features_test.toArray.flatten[Byte])

    val labels_tensor_test = dtf.tensor_from("FLOAT32", test_set.length, 2)(labels_test.flatten[Double])

    println("Helios data set created")
    working_set.copy(
      trainData   = features_tensor_train,
      trainLabels = labels_tensor_train,
      testData    = features_tensor_test,
      testLabels  = labels_tensor_test
    )
  }

  /**
    * Create a processed tensor data set as a [[HeliosDataSet]] instance.
    *
    * @param collated_data A Stream of date times, image paths and fluxes.
    *
    * @param tt_partition A function which takes each data element and
    *                     determines if it goes into the train or test split.
    *
    * @param scaleDownFactor The exponent of 2 which determines how much the
    *                        image will be scaled down. i.e. scaleDownFactor = 4
    *                        corresponds to a 16 fold decrease in image size.
    * */
  def create_helios_goes_data_set(
    collated_data: Stream[(DateTime, (Path, (Double, Double)))],
    tt_partition: ((DateTime, (Path, (Double, Double)))) => Boolean,
    scaleDownFactor: Int = 4, resample: Boolean = false): HeliosDataSet = {

    val scaleDown = 1/math.pow(2, scaleDownFactor)

    print("Scaling down images by a factor of ")
    pprint.pprintln(math.pow(2, scaleDownFactor))
    println()

    println("Separating data into train and test.")
    val (train_set, test_set) = collated_data.partition(tt_partition)

    //Calculate the height, width and number of channels
    //in the images
    val (scaled_height, scaled_width, num_channels) = {

      val im = Image.fromPath(train_set.head._2._1.toNIO)

      val scaled_image = im.copy.scale(scaleDown)

      (scaled_image.height, scaled_image.width, scaled_image.argb(0, 0).length)

    }

    val working_set = HeliosDataSet(
      null, null, train_set.length,
      null, null, test_set.length)

    /*
    * If the `resample` flag is set to true,
    * balance the occurence of high and low
    * flux events through re-sampling.
    *
    * */
    val processed_train_set = if(resample) {
      //Resample training set ot
      //emphasize extreme events.

      val un_prob = train_set.map(_._2._2._1).map(math.exp)
      val normalizer = un_prob.sum
      val selector = MultinomialRV(DenseVector(un_prob.toArray)/normalizer)

      helios.resample(train_set, selector)
    } else train_set

    val (features_train, labels_train): (Stream[Array[Byte]], Stream[Seq[Double]]) =
      processed_train_set.map(entry => {
        val (_, (path, data_label)) = entry

        val im = Image.fromPath(path.toNIO)

        val scaled_image = im.copy.scale(scaleDown)

        (scaled_image.argb.flatten.map(_.toByte), Seq(data_label._1, data_label._2))

      }).unzip

    val features_tensor_train = dtf.tensor_from_buffer(
      "UINT8", processed_train_set.length, scaled_height, scaled_width, num_channels)(
      features_train.toArray.flatten[Byte])

    val labels_tensor_train = dtf.tensor_from("FLOAT32", train_set.length, 2)(labels_train.flatten[Double])


    val (features_test, labels_test): (Stream[Array[Byte]], Stream[Seq[Double]]) = test_set.map(entry => {
      val (_, (path, data_label)) = entry

      val im = Image.fromPath(path.toNIO)

      val scaled_image = im.copy.scale(scaleDown)

      (scaled_image.argb.flatten.map(_.toByte), Seq(data_label._1, data_label._2))

    }).unzip

    val features_tensor_test = dtf.tensor_from_buffer(
      "UINT8", test_set.length, scaled_height, scaled_width, num_channels)(
      features_test.toArray.flatten[Byte])

    val labels_tensor_test = dtf.tensor_from("FLOAT32", test_set.length, 2)(labels_test.flatten[Double])

    println("Helios data set created")
    working_set.copy(
      trainData   = features_tensor_train,
      trainLabels = labels_tensor_train,
      testData    = features_tensor_test,
      testLabels  = labels_tensor_test
    )
  }

  /**
    * Calculate RMSE of a tensorflow based estimator.
    * */
  def calculate_rmse(
    n: Int, n_part: Int)(
    labels_mean: Tensor, labels_stddev: Tensor)(
    images: Tensor, labels: Tensor)(infer: (Tensor) => Tensor): Float = {

    def accuracy(im: Tensor, lab: Tensor): Float = {
      infer(im)
        .multiply(labels_stddev)
        .add(labels_mean)
        .subtract(lab).cast(FLOAT32)
        .square.mean().scalar
        .asInstanceOf[Float]
    }

    val num_elem: Int = n/n_part

    math.sqrt((0 until n_part).map(i => {

      val (lower_index, upper_index) = (i*num_elem, if(i == n_part-1) n else (i+1)*num_elem)

      accuracy(images(lower_index::upper_index, ::, ::, ::), labels(lower_index::upper_index))
    }).sum/num_elem).toFloat
  }

  /**
    * Generate a starting data set for GOES prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]] method.
    *
    * @param image_source The [[SOHO]] data source to extract from
    * @param year_start The starting time of the data
    * @param year_end The end time of the data.
    * */
  def generate_data_goes(
    year_start: Int = 2001, year_end: Int = 2005,
    image_source: SOHO = SOHO(SOHOData.Instruments.MDIMAG, 512))
  : Stream[(DateTime, (Path, (Double, Double)))] = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: "+os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: "+user_name)

    val home_dir_prefix = if(os_name.startsWith("Mac")) root/"Users" else root/'home

    require(year_end > year_start, "Data set must encompass more than one year")

    /*
    * Create a collated data set,
    * extract GOES flux data and join it
    * with eit195 (green filter) images.
    * */

    print("Looking for data in directory ")
    val data_dir = home_dir_prefix/user_name/"data_repo"/'helios
    pprint.pprintln(data_dir)

    val soho_dir = data_dir/'soho
    val goes_dir = data_dir/'goes

    val reduce_fn = (gr: Stream[(DateTime, (Double, Double))]) => {

      val max_flux = gr.map(_._2).max

      (gr.head._1, (math.log10(max_flux._1), math.log10(max_flux._2)))
    }

    val round_date = (d: DateTime) => {

      val num_minutes = 10

      val minutes: Int = d.getMinuteOfHour/num_minutes

      new DateTime(
        d.getYear, d.getMonthOfYear,
        d.getDayOfMonth, d.getHourOfDay,
        minutes*num_minutes)
    }

    println("Preparing data-set as a Stream ")
    println("Start: "+year_start+" End: "+year_end)


    helios.collate_goes_data_range(
      new YearMonth(year_start, 1), new YearMonth(year_end, 12))(
      GOES(GOESData.Quantities.XRAY_FLUX_5m),
      goes_dir,
      goes_aggregation = 2,
      goes_reduce_func = reduce_fn,
      image_source,
      soho_dir,
      dt_round_off = round_date)

  }

  /**
    * Generate a starting data set for OMNI/L1 prediction tasks.
    * This method makes the assumption that the data is stored
    * in a directory ~/data_repo/helios in a standard directory tree
    * generated after executing the [[data.SOHOLoader.bulk_download()]] method.
    *
    * @param image_source The [[SOHO]] data source to extract from
    * @param year_start The starting time of the data
    * @param year_end The end time of the data.
    * */
  def generate_data_omni(
    year_start: Int = 2001, year_end: Int = 2005,
    image_source: SOHO = SOHO(SOHOData.Instruments.MDIMAG, 512),
    omni_source: OMNI = OMNI(OMNIData.Quantities.V_SW))
  : Stream[(DateTime, (Path, Seq[Double]))] = {

    /*
     * Mind your surroundings!
     * */
    val os_name = System.getProperty("os.name")

    println("OS: "+os_name)

    val user_name = System.getProperty("user.name")

    println("Running as user: "+user_name)

    val home_dir_prefix = if(os_name.startsWith("Mac")) root/"Users" else root/'home

    require(year_end > year_start, "Data set must encompass more than one year")

    print("Looking for data in directory ")
    val data_dir = home_dir_prefix/user_name/"data_repo"/'helios
    pprint.pprintln(data_dir)

    val soho_dir = data_dir/'soho

    println("Preparing data-set as a Stream ")
    println("Start: "+year_start+" End: "+year_end)


    helios.collate_omni_data_range(
      new YearMonth(year_start, 1), new YearMonth(year_end, 12))(
      omni_source, pwd/"data", (12, 72), image_source, soho_dir)
  }


  /**
    * Train a Neural architecture on a
    * processed data set.
    *
    * @param collated_data Data set of temporally joined
    *                      image paths and GOES X-Ray fluxes.
    *                      This is generally the output after
    *                      executing [[collate_goes_data_range()]]
    *                      with the relevant parameters.
    * @param tt_partition A function which splits the data set
    *                     into train and test sections, based on
    *                     any Boolean function. If the function
    *                     returns true then the instance falls into
    *                     the training set else the test set
    * @param resample If set to true, the training data is resampled
    *                 to balance the occurrence of high flux and low
    *                 flux events.
    * @param longWavelength If set to true, predict long wavelength
    *                       GOES X-Ray flux, else short wavelength,
    *                       defaults to short wavelength.
    * @param tempdir A working directory where the results will be
    *                archived, defaults to user_home_dir/tmp. The model
    *                checkpoints and other results will be stored inside
    *                another directory created in tempdir.
    * @param results_id The suffix added the results/checkpoints directory name.
    * @param max_iterations The maximum number of iterations that the
    *                       network must be trained for.
    * @param arch The neural architecture to train, defaults to [[Arch.cnn_goes_v1]]
    *
    * */
  def run_experiment_goes(
    collated_data: Stream[(DateTime, (Path, (Double, Double)))],
    tt_partition: ((DateTime, (Path, (Double, Double)))) => Boolean,
    resample: Boolean = false, longWavelength: Boolean = false)(
    results_id: String, max_iterations: Int,
    tempdir: Path = home/"tmp",
    arch: Layer[Output, Output] = learn.cnn_goes_v1,
    lossFunc: Loss[(Output, Output)] = tf.learn.L2Loss("Loss/L2")) = {

    val resDirName = if(longWavelength) "helios_goes_long_"+results_id else "helios_goes_"+results_id

    val tf_summary_dir = tempdir/resDirName

    val checkpoints =
      if (exists! tf_summary_dir) ls! tf_summary_dir |? (_.isFile) |? (_.segments.last.contains("model.ckpt-"))
      else Seq()

    val checkpoint_max =
      if(checkpoints.isEmpty) 0
      else (checkpoints | (_.segments.last.split("-").last.split('.').head.toInt)).max

    val iterations = if(max_iterations > checkpoint_max) max_iterations - checkpoint_max else 0


    /*
    * After data has been joined/collated,
    * start loading it into tensors
    *
    * */

    val dataSet = helios.create_helios_goes_data_set(
      collated_data,
      tt_partition,
      scaleDownFactor = 2,
      resample)

    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainData)

    val targetIndex = if(longWavelength) 1 else 0

    val train_labels = dataSet.trainLabels(::, targetIndex)

    val labels_mean = train_labels.mean()

    val labels_stddev = train_labels.subtract(labels_mean).square.mean().sqrt

    val norm_train_labels = train_labels.subtract(labels_mean).divide(labels_stddev)

    val trainLabels = tf.data.TensorSlicesDataset(norm_train_labels)

    //val trainWeights = tf.data.TensorSlicesDataset(norm_train_labels.exp)

    val trainData =
      trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(64)
        .prefetch(10)

    /*
    * Start building tensorflow network/graph
    * */
    println("Building the regression model.")
    val input = tf.learn.Input(
      UINT8,
      Shape(
        -1,
        dataSet.trainData.shape(1),
        dataSet.trainData.shape(2),
        dataSet.trainData.shape(3))
    )

    val trainInput = tf.learn.Input(FLOAT32, Shape(-1))

    val trainingInputLayer = tf.learn.Cast("TrainInput", INT64)

    val loss = lossFunc >>
      tf.learn.Mean("Loss/Mean") >>
      tf.learn.ScalarSummary("Loss", "ModelLoss")

    val optimizer = tf.train.AdaGrad(0.002)

    val summariesDir = java.nio.file.Paths.get(tf_summary_dir.toString())

    //Now create the model
    val (model, estimator) = tf.createWith(graph = Graph()) {
      val model = tf.learn.Model(
        input, arch, trainInput, trainingInputLayer,
        loss, optimizer)

      println("Training the linear regression model.")

      val estimator = tf.learn.FileBasedEstimator(
        model,
        tf.learn.Configuration(Some(summariesDir)),
        tf.learn.StopCriteria(maxSteps = Some(iterations)),
        Set(
          tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(5000)),
          tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(5000)),
          tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(5000))),
        tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 5000))

      estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(iterations)))

      (model, estimator)
    }


    val accuracy = helios.calculate_rmse(dataSet.nTest, 4)(labels_mean, labels_stddev) _

    val testAccuracy = accuracy(
      dataSet.testData, dataSet.testLabels(::, targetIndex))(
      (im: Tensor) => estimator.infer(() => im))

    print("Test accuracy = ")
    pprint.pprintln(testAccuracy)

    dataSet.close()

    (model, estimator, testAccuracy, tf_summary_dir, labels_mean, labels_stddev)
  }



}
