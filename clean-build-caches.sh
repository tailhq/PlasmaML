#! /bin/bash

echo 'Cleaning local Ivy cache'
rm -rf ~/.ivy/*

echo 'Cleaning local coursier caches (if any)'
rm -rf ~/.coursier/*
rm -rf ~/Library/Caches/Coursier/*