# PlasmaML

[![Build Status](https://travis-ci.org/transcendent-ai-labs/PlasmaML.svg?branch=master)](https://travis-ci.org/transcendent-ai-labs/PlasmaML)

Machine Learning tools for Space Weather and Plasma Physics
---------------------------
> ![Image courtesy NASA](http://www.nasa.gov/images/content/607990main1_FAQ13-670.jpg)
>
> courtesy [NASA](www.nasa.gov)

*PlasmaML* is a collection of data analysis and machine learning tools in the domain of space physics, more specifically in modelling of space plasmas & space weather prediction.

This is a multi-language project where the primary modelling is done in *Scala* while *R* is heavily leveraged for generating visualizations. The project depends on the [DynaML](https://github.com/mandar2812/DynaML) scala machine learning library and uses model and optimization implementations in it as a starting point for extensive experiments in space physics simulations and space weather prediction.

## Getting Started

*PlasmaML* is managed using the Simple Build Tool (sbt).

### Installation

#### Requirements

1. Java Development Kit 8. 
2. [Scala](scala-lang.org)
3. [sbt](http://www.scala-sbt.org/)
4. [R](https://www.r-project.org/) with the following packages:

    * `ggplot2`
    * `reshape2`
    * `latex2exp`
    * `plyr`
    * `gridExtra`
    * `reshape2`
    * `directlabels`


#### Steps

After cloning the project, PlasmaML can be installed directly from the shell or 
by first entering the sbt shell and building the source.

**From the shell**

From the root directory `PlasmaML` run the build script (with configurable parameters).

```bash
./build.sh <heap size> <compile with gpu support> <use packaged tensorflow> <update bash env>
```

For example the following builds the project with 4 GB java heap and GPU support.

```bash
./build.sh 4096m true
```

Note that for Nvidia GPU support to work, compatible versions of CUDA and cuDNN must be installed and 
found in the `$LD_LIBRARY_PATH` environment variable see the [DynaML docs](https://transcendent-ai-labs.github.io/DynaML/installation/installation/) for more info.

Use the last parameter `<update bash env>` to add the PlasmaML executable in the bash `$PATH`.

The following build will use 4 GB of heap, with GPU support, precompiled tensorflow binaries and 
adds `plasmaml` binary to the `$PATH` variable.

```
./build.sh 4096m true false true
```

**From the sbt shell**

Start the sbt shell with the script `sbt-shell.sh` having the same parameters as `build.sh`

```bash
./build.sh <heap size> <compile with gpu support> <use packaged tensorflow>
```

From the sbt shell, run

```
stage
```

After building, access the PlasmaML shell like 

```
./target/universal/stage/bin/plasmaml
```