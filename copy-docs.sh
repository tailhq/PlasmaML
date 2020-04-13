#! /bin/bash

mkdir -p ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1
mkdir -p ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/omni
mkdir -p ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/mag-core
mkdir -p ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/helios
mkdir -p ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/streamer
mkdir -p ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/vanAllen

sbt stage

cp -R omni/target/scala-2.12/api/* ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/omni/
cp -R mag-core/target/scala-2.12/api/* ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/mag-core/
cp -R helios/target/scala-2.12/api/* ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/helios/
cp -R streamer/target/scala-2.12/api/* ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/streamer/
cp -R vanAllen/target/scala-2.12/api/* ../transcendent-ai-labs.github.io/api_docs/PlasmaML/$1/vanAllen/
