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

lazy val root = (project in file(".")).settings(commonSettings: _*).aggregate(omni)

lazy val omni = (project in file("omni")).settings(commonSettings: _*).settings(
    initialCommands in console :=
      """import io.github.mandar2812.PlasmaML.omni._;"""+
      """import io.github.mandar2812.dynaml.kernels._"""
)