package io.github.mandar2812.PlasmaML.helios

import ammonite.ops.Path
import io.github.mandar2812.dynaml.utils

package object data {

  /**
    * Download a resource (image, file) from a sequence of urls to a specified
    * disk location.
    * */
  def download_batch(path: Path)(urls: List[String]): Unit = {
    urls.par.foreach(s => utils.downloadURL(s, (path/s.split('/').last).toString()))
  }
}
