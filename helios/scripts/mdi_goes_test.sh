#!/usr/bin/env bash

yeartest=$1
limit=$2
mdiscript="./helios/scripts/goes_extreme_events.sc"

while [ "$yeartest" -lt "$limit" ]; do
./target/universal/stage/bin/plasmaml $mdiscript $yeartest
let yeartest+=1
done
