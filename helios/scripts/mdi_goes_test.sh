#!/usr/bin/env bash
yeartest=2001
limit=2006

while [ "$yeartest" -lt "$limit" ]; do
./target/universal/stage/bin/plasmaml $"yeartest"
let yeartest+=1
done
