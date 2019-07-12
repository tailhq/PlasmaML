#! /bin/bash

id=${1:-csss_exp1}

year=${2:-2015}

month=${3:-10}

tr=${4:-2077}

./target/universal/stage/bin/plasmaml ./helios/scripts/csss_job.sc --csss_job_id $id --test_year $year --test_month $month --test_rotation $tr
