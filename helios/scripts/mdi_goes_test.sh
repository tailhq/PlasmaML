#!/usr/bin/env bash

yeartest=$2
limit=$3
mdiscript=$1

while [ "$yeartest" -lt "$limit" ]; do
./target/universal/stage/bin/plasmaml $mdiscript $yeartest
let yeartest+=1
done
