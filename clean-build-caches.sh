#! /bin/bash

echo 'Cleaning local Ivy cache'
rm -rf ~/.ivy/*

echo 'Cleaning local coursier caches (if any)'
rm ~/.coursier/*
rm ~/Library/Caches/Coursier/*