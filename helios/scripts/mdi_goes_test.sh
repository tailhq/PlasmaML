#!/usr/bin/env bash
year=2001

while [ $(year) -lt 2006 ]; do
./target/universal/stage/bin/plasmaml $(year)
let year+=1
done
