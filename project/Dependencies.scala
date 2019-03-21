import java.io.File

import sbt._

object Dependencies {

  val scala_major = 2.12

  val scala_minor = 4

  val scala = s"$scala_major.$scala_minor"

  val crossScala = Seq("2.12.4")

  val tfscala_version = "0.4.2-SNAPSHOT"

  private def process_flag(s: String) = if(s.toLowerCase == "true" || s == "1") true else false

  //Set to true if, building with Nvidia GPU support.
  val gpuFlag: Boolean = process_flag(Option(System.getProperty("gpu")).getOrElse("false"))

  //Set to false if using self compiled tensorflow library
  val packagedTFFlag: Boolean = true

  //Set to dev, if pulling DynaML master SNAPSHOT
  val status = "dev"

  val dataDirectory = settingKey[File]("The directory holding the data files for running example scripts")

  val latest_dynaml_release = "v1.5.3"
  val latest_dynaml_dev_release = "v2.0-tf-0.4.x"
  val dynaml_branch = ""


  val (dynamlGroupID, dynamlArtifact, dynaMLVersion) =
    if(status == "local") ("io.github.transcendent-ai-labs", "dynaml", s"$latest_dynaml_dev_release-SNAPSHOT")
    else if(status == "dev") ("io.github.transcendent-ai-labs", "dynaml", s"$latest_dynaml_dev_release-SNAPSHOT")
    else ("io.github.transcendent-ai-labs", "dynaml", latest_dynaml_release)

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
    "org.jsoup" % "jsoup" % "1.9.1",
    "joda-time" % "joda-time" % "2.10",
    "org.joda" % "joda-convert" % "2.1",
    "org.json4s" %% "json4s-native" % "3.6.2",
    "com.typesafe.slick" %% "slick" % "3.2.3"
  )

  val dynaMLDependency = Seq(dynamlGroupID %% dynamlArtifact % dynaMLVersion)
    .map(_.withExclusions(
      Vector(
        "org.platanios" %% "tensorflow", 
        "org.platanios" %% "tensorflow-data",
        "org.platanios" %% "tensorflow-jni",
        "org.platanios" %% "tensorflow-examples",
        "org.platanios" %% "tensorflow-api",
        "org.platanios" %% "tensorflow-horovod"
      )))

  val tf =
    if(packagedTFFlag) "org.platanios" %% "tensorflow" % tfscala_version classifier tensorflow_classifier
    else "org.platanios" %% "tensorflow" % tfscala_version

  val tf_examples = "org.platanios" %% "tensorflow-data" % tfscala_version

  val tensorflowDependency = Seq(
    tf,
    tf_examples
  ).map(_.withExclusions(Vector("org.typelevel" %% "spire")))
}
