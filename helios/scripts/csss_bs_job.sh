#! /bin/bash

id=${1:-csss_exp1}

dir=${2}

root=${3:-$HOME/tmp}

./target/universal/stage/bin/plasmaml ./helios/scripts/csss_bs_job.sc --csss_job_id $id --exp_dir $dir --root_dir $root
