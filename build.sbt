import sbt._

val dynaMLVersion = settingKey[String]("The version of DynaML used.")

lazy val commonSettings = Seq(
  name := "PlasmaML",
  organization := "io.github.mandar2812",
  version := "0.1.0",
  scalaVersion in ThisBuild := "2.11.8",
  dynaMLVersion := "master-SNAPSHOT",
  libraryDependencies in ThisBuild ++= Seq(
    "com.github.mandar2812" % "DynaML" % dynaMLVersion.value,
    "org.jsoup" % "jsoup" % "1.9.1"
  )
)

resolvers in ThisBuild ++= Seq(
  "jitpack" at "https://jitpack.io",
  "jzy3d-releases" at "http://maven.jzy3d.org/releases"
)

lazy val root = (project in file(".")).settings(commonSettings: _*)
  .aggregate(omni, vanAllen)
  .settings(
    aggregate in update := false
  )

lazy val omni = (project in file("omni")).settings(commonSettings: _*).settings(
  initialCommands in console :=
  """import io.github.mandar2812.PlasmaML.omni._;"""+
    """import io.github.mandar2812.dynaml.kernels._;"""+
    """import com.quantifind.charts.Highcharts._"""
)

lazy val vanAllen = (project in file("vanAllen")).settings(commonSettings: _*).settings(
  initialCommands in console :=
    """import io.github.mandar2812.PlasmaML.vanAllen._;"""+
      """import io.github.mandar2812.dynaml.kernels._;"""+
      """import com.quantifind.charts.Highcharts._;"""+
      """import org.jsoup._"""
)