#! /bin/bash

gpuFlag=${2:-false}

heapSize=${1:-4096m}

echo "Commencing PlasmaML Build with: Executable Heap Size = $heapSize and GPU Flag = $gpuFlag"

sbt -Dheap=${heapSize} -Dgpu=${gpuFlag} stage

chmod +x ./target/universal/stage/bin/plasmaml

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
        bash_file=".bashrc"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        bash_file=".bash_profile"
elif [[ "$OSTYPE" == "cygwin" ]]; then
        # POSIX compatibility layer and Linux environment emulation for Windows
        bash_file=".bashrc"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
        bash_file=".bash_profile"
else
        bash_file=".bashrc"
fi

echo "Updating PLASMAML_HOME=$DIR variable in $bash_file"

sed -i.bak '/export PLASMAML_HOME/d' ~/${bash_file}
echo 'export PLASMAML_HOME='${DIR} >>~/${bash_file}
source ~/${bash_file}