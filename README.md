# PlasmaML

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

**Requirements**

1. [Scala](scala-lang.org)
2. [sbt](http://www.scala-sbt.org/)
3. [R](https://www.r-project.org/)

**Steps**

After installing all the pre-requisites,

1. Clone this project
2. From the root directory type ```sbt```, you should see the sbt prompt.
4. At the sbt prompt, choose appropriate sub-project ```project omni```
5. Compile the sources ```compile```
6. Run the scala console ```console```

Try an example program 

```scala
scala>TestOmniNarmax("2004/11/09/08", "2004/11/11/09", "predict")
```

