import java.io.File

import sbt._
import Dependencies._

val mainVersion = "v0.1"

lazy val commonSettings = Seq(
  organization := "io.github.mandar2812",
  scalaVersion in ThisBuild := scala,
  libraryDependencies in ThisBuild ++= (commonDependencies ++ dynaMLDependency ++ tensorflowDependency),
  resolvers in ThisBuild ++= Seq(
    "jzy3d-releases" at "http://maven.jzy3d.org/releases",
    "Scalaz Bintray Repo" at "http://dl.bintray.com/scalaz/releases",
    "BeDataDriven" at "https://nexus.bedatadriven.com/content/groups/public",
    "atlassian-maven" at "https://maven.atlassian.com/maven-external",
    Resolver.sonatypeRepo("public"),
    Resolver.sonatypeRepo("snapshots")
  )
)

lazy val mag_core = (project in file("mag-core")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
  .settings(commonSettings: _*)
  .settings(
    initialCommands in console :=
      """import io.github.mandar2812.PlasmaML._;"""+
        """import io.github.mandar2812.PlasmaML.cdf.CDFUtils;"""+
        """import scalaxy.streams.optimize;"""+
        """import io.github.mandar2812.dynaml.kernels._;"""+
        """import io.github.mandar2812.dynaml.DynaMLPipe;"""+
        """import com.quantifind.charts.Highcharts._;"""+
        """import breeze.linalg.DenseVector;""" ,
    scalacOptions ++= Seq("-optimise", "-Yclosure-elim", "-Yinline"))


lazy val omni =
  (project in file("omni")).enablePlugins(JavaAppPackaging, BuildInfoPlugin).settings(commonSettings: _*)
    .settings(
      initialCommands in console :=
        """import io.github.mandar2812.PlasmaML.omni._;"""+
          """import scalaxy.streams.optimize;"""+
          """import io.github.mandar2812.dynaml.kernels._;"""+
          """import io.github.mandar2812.dynaml.DynaMLPipe;"""+
          """import com.quantifind.charts.Highcharts._;"""+
          """import breeze.linalg.DenseVector;"""
    ).dependsOn(mag_core)

lazy val vanAllen =
  (project in file("vanAllen")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
    .settings(commonSettings: _*)
    .settings(
      initialCommands in console :=
        """import io.github.mandar2812.PlasmaML.vanAllen._;"""+
          """import org.json4s._;"""+
          """import org.json4s.jackson.JsonMethods._;"""+
          """import io.github.mandar2812.dynaml.kernels._;"""+
          """import io.github.mandar2812.dynaml.pipes._;"""+
          """import com.quantifind.charts.Highcharts._;"""+
          """import org.jsoup._;"""+
          """import breeze.linalg.{DenseMatrix, DenseVector};"""
    ).dependsOn(mag_core)

lazy val streamer =
  (project in file("streamer")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
    .settings(commonSettings: _*)
    .settings(
      initialCommands in console :=
        """import io.github.mandar2812.PlasmaML.streamer._;"""+
        """import io.github.mandar2812.dynaml.kernels._;"""+
          """import io.github.mandar2812.dynaml.DynaMLPipe;"""+
          """import com.quantifind.charts.Highcharts._;"""+
          """import breeze.linalg.DenseVector;"""
    ).dependsOn(mag_core)

lazy val helios =
  (project in file("helios")).enablePlugins(JavaAppPackaging, BuildInfoPlugin)
    .settings(commonSettings: _*)
    .settings(
      initialCommands in console :=
        """import io.github.mandar2812.dynaml.kernels._;"""+
          """import io.github.mandar2812.dynaml.DynaMLPipe;"""+
          """import com.quantifind.charts.Highcharts._;"""+
          """import breeze.linalg.DenseVector;"""
    ).dependsOn(mag_core, omni)

lazy val PlasmaML = (project in file(".")).enablePlugins(JavaAppPackaging, BuildInfoPlugin, sbtdocker.DockerPlugin)
  .settings(commonSettings: _*)
  .dependsOn(mag_core, omni, vanAllen, streamer, helios).settings(
  name := "PlasmaML",
  version := mainVersion,
  fork in run := true,
  mainClass in Compile := Some("io.github.mandar2812.PlasmaML.PlasmaML"),
  buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
  buildInfoPackage := "io.github.mandar2812.PlasmaML",
  buildInfoUsePackageAsPath := true,
  mappings in Universal ++= Seq({
    // we are using the reference.conf as default application.conf
    // the user can override settings here
    val init = (resourceDirectory in Compile).value / "DynaMLInit.scala"
    init -> "conf/DynaMLInit.scala"
  }, {
    val banner = (resourceDirectory in Compile).value / "banner.txt"
    banner -> "conf/banner.txt"
  }),
  javaOptions in Universal ++= Seq(
    // -J params will be added as jvm parameters
    "-J-Xmx4096m",
    "-J-Xms64m"
  ),
  dataDirectory := new File("data/"),
  initialCommands in console :="""io.github.mandar2812.PlasmaML.PlasmaML.main(Array())""",
  dockerfile in docker := {
    val appDir: File = stage.value
    val targetDir = "/app"

    new Dockerfile {
      from("openjdk:8-jre")
      entryPoint(s"$targetDir/bin/${executableScriptName.value}")
      copy(appDir, targetDir, chown = "daemon:daemon")
    }
  }
).aggregate(mag_core, omni, vanAllen, streamer, helios)
  .settings(aggregate in update := false)

