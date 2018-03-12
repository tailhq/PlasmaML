import java.io.File

import sbt._

object Dependencies {

  val scala = "2.11.8"

  val tfscala_version = "0.1.1"

  //Set to true if, building with Nvidia GPU support.
  val gpuFlag: Boolean = false

  //Set to false if using self compiled tensorflow library
  val packagedTFFlag: Boolean = true

  //Set to dev, if pulling DynaML master SNAPSHOT
  val status = "dev"

  val dataDirectory = settingKey[File]("The directory holding the data files for running example scripts")

  val latest_dynaml_release = "v1.5.3-beta.1"

  val (dynamlGroupID, dynamlArtifact, dynaMLVersion) =
    if(status == "local") ("io.github.mandar2812", "dynaml_2.11", latest_dynaml_release)
    else if(status == "dev") ("com.github.transcendent-ai-labs.DynaML", "dynaml_2.11", "-SNAPSHOT")
    else ("com.github.transcendent-ai-labs.DynaML", "dynaml_2.11", latest_dynaml_release)

  val platform: String = {
    // Determine platform name using code similar to javacpp
    // com.googlecode.javacpp.Loader.java line 60-84
    val jvmName = System.getProperty("java.vm.name").toLowerCase
    var osName = System.getProperty("os.name").toLowerCase
    var osArch = System.getProperty("os.arch").toLowerCase
    if (jvmName.startsWith("dalvik") && osName.startsWith("linux")) {
      osName = "android"
    } else if (jvmName.startsWith("robovm") && osName.startsWith("darwin")) {
      osName = "ios"
      osArch = "arm"
    } else if (osName.startsWith("mac os x")) {
      osName = "macosx"
    } else {
      val spaceIndex = osName.indexOf(' ')
      if (spaceIndex > 0) {
        osName = osName.substring(0, spaceIndex)
      }
    }
    if (osArch.equals("i386") || osArch.equals("i486") || osArch.equals("i586") || osArch.equals("i686")) {
      osArch = "x86"
    } else if (osArch.equals("amd64") || osArch.equals("x86-64") || osArch.equals("x64")) {
      osArch = "x86_64"
    } else if (osArch.startsWith("arm")) {
      osArch = "arm"
    }
    val platformName = osName + "-" + osArch
    println("platform: " + platformName)
    platformName
  }

  val tensorflow_classifier: String = {
    val platform_splits = platform.split("-")
    val (os, arch) = (platform_splits.head, platform_splits.last)

    val tf_c =
      if (os.contains("macosx")) "darwin-cpu-"+arch
      else if(os.contains("linux")) {
        if(gpuFlag) "linux-gpu-"+arch else "linux-cpu-"+arch
      } else ""
    println("Tensorflow-Scala Classifier: "+tf_c)
    tf_c
  }

  val commonDependencies = Seq(
    "com.nativelibs4java" % "scalaxy-streams_2.11" % "0.3.4" % "provided",
    "org.jsoup" % "jsoup" % "1.9.1",
    "joda-time" % "joda-time" % "2.9.3",
    "org.json4s" % "json4s-native_2.11" % "3.3.0",
    "com.typesafe.slick" %% "slick" % "3.1.1"
  )

  val dynaMLDependency = Seq(
    dynamlGroupID % dynamlArtifact % dynaMLVersion)
    .map(_.exclude("org.platanios", "tensorflow_2.11"))
    .map(_.exclude("org.platanios", "tensorflow-data_2.11"))

  val tf =
    if(packagedTFFlag) "org.platanios" % "tensorflow_2.11" % tfscala_version classifier tensorflow_classifier
    else "org.platanios" % "tensorflow_2.11" % tfscala_version

  val tf_examples = "org.platanios" % "tensorflow-data_2.11" % tfscala_version

  val tensorflowDependency = Seq(
    tf,
    tf_examples
  ).map(_.exclude("org.typelevel", "spire_2.11"))
}
