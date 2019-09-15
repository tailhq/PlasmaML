#! /bin/bash

id=${2:-dtlr_cv_sw}

root=${1}

echo "Looking in $root/$id for existing CSSS runs."

count=0

nl=$(ls -dq $root/$id/fte_omni_mo_tl* | wc -l)


for path in $root/$id/fte_omni_mo_tl*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"
    echo "Found run: $dirname"
    echo "Training base line solar wind predictor."
    ./target/universal/stage/bin/plasmaml ./helios/scripts/csss_bs_job.sc --csss_job_id $id --exp_dir $dirname --root_dir $root
    if [ $? -eq 0 ]; then
        count=$((count + 1))
    fi
done

echo "Successfully ran $count out of $nl folds."
