#! /bin/bash

filepath=${2:-"data/sw_cv_dtlr.txt"}

id=${1:-dtlr_cv_sw}

count=0

nl=$(sed -n '$=' $filepath)

echo "Solar Wind Cross Validation: $nl fold."
echo "Experiment Id: $id"

while read tr; do
    echo "Running experiment for Carrington rotation $tr"
    ./target/universal/stage/bin/plasmaml ./helios/scripts/csss_job.sc --csss_job_id $id --test_rotation $tr
    if [ $? -eq 0 ]; then
        count=$((count + 1))
    fi
done <$filepath

echo "Successfully ran $count out of $nl folds."