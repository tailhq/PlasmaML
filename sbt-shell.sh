#! /bin/bash

gpuFlag=${2:-false}

heapSize=${1:-4096m}

pTF=${3:-false}

echo "Starting sbt shell with Build Executable Heap Size = $heapSize and GPU Flag = $gpuFlag"

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag} -DpackagedTF=${pTF}