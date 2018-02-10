#!/usr/bin/env bash

yeartest=$2
limit=$3
mdiscript=$1
longwv="true"

while [ "$yeartest" -lt "$limit" ]; do
./target/universal/stage/bin/plasmaml $mdiscript $yeartest $longwv
let yeartest+=1
done
