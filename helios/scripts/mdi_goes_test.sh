#!/usr/bin/env bash

yeartest=2001
limit=2006
mdiscript="./helios/scripts/goes_extreme_events.sc"

while [ "$yeartest" -lt "$limit" ]; do
exec ./target/universal/stage/bin/plasmaml "$mdiscript" "$yeartest"
let yeartest+=1
done

!#