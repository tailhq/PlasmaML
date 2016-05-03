import sbt._

lazy val commonSettings = Seq(
  name := "PlasmaML",
  organization := "io.github.mandar2812",
  version := "0.1.0",
  scalaVersion := "2.11.8"
)

scalaVersion in ThisBuild := "2.11.8"

resolvers in ThisBuild ++= Seq("jitpack" at "https://jitpack.io", "jzy3d-releases" at "http://maven.jzy3d.org/releases")

libraryDependencies in ThisBuild += "com.github.mandar2812" % "DynaML" % "master-SNAPSHOT"

initialCommands in console := "import breeze.linalg._,import io.github.mandar2812.dynaml.models._,"+
  "import io.github.mandar2812.dynaml.models.neuralnets._,import io.github.mandar2812.dynaml.models.svm._,"+
  "import io.github.mandar2812.dynaml.models.lm._,import io.github.mandar2812.dynaml.utils,"+
  "import io.github.mandar2812.dynaml.kernels._,"+
  "import org.apache.spark.{SparkContext,SparkConf},"+
  "import io.github.mandar2812.dynaml.pipes._,import org.openml.apiconnector.io._"

lazy val root = (project in file(".")).settings(commonSettings: _*).aggregate(omni)

lazy val omni = (project in file("omni")).settings(commonSettings: _*)