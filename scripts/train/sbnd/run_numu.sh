#!/bin/bash

usage ()
{
    cat <<EOF
USAGE: run_numu.sh TAG TRAIN_DIR...

Full SBND numu CC pipeline, from WireCell flat CAFs to an evaluated model:
  1. eval set      \$VLNE_DATADIR/sbnd/eval_numu.hdf (built from EVAL_DIR if missing)
  2. training set  \$VLNE_DATADIR/sbnd/train_numu_TAG.hdf from every
                   *.wc.flat.caf.root in the TRAIN_DIRs (and their subdirectories)
  3. check that no training file is also an eval file
  4. shuffle       \$VLNE_DATADIR/sbnd/train_numu_TAG_shuffled.hdf
  5. config        scripts/train/sbnd/numu_TAG.py, a copy of BASE_CONFIG
  6. train         \$VLNE_OUTDIR/sbnd/numu_TAG/model_m(lstm_v2)_*
  7. eval          on the test split of the eval set

Steps whose output exists are skipped, so a rerun resumes where it stopped.
Delete an output to redo that step (and the ones after it).

Run with the vlne env active (conda activate vlne), from any directory.
Environment overrides:
  EVAL_DIR     default ~/Downloads/haiwang_stuff
  BASE_CONFIG  default scripts/train/sbnd/numu_v0.py
  JOBS         converter processes, default 8

Example:
  scripts/train/sbnd/run_numu.sh v2 ~/RNN_energy/flatcafs \\
      ~/Downloads/haiwang_stuff_501_1000 ~/Downloads/new_batch
EOF
    exit 1
}

die ()
{
    echo "$@" >&2
    exit 1
}

step ()
{
    echo -e "\n=== $*"
}

summary ()
{
    cut -d, -f1-3,6 "${1}" | column -t -s,
}

[[ "${#}" -lt 2 ]] && usage
tag="${1}"
shift
train_dirs=( "${@}" )

EVAL_DIR="${EVAL_DIR:-${HOME}/Downloads/haiwang_stuff}"
BASE_CONFIG="${BASE_CONFIG:-scripts/train/sbnd/numu_v0.py}"
JOBS="${JOBS:-8}"

cd "$(dirname "${0}")/../../.." || die "Cannot cd to the vlne repository"

# Not plain `python`: ~/.zshrc aliases it to the system python3.13
python="${CONDA_PREFIX}/bin/python"
[[ -n "${VLNE_DATADIR}" && -n "${VLNE_OUTDIR}" ]] \
    && "${python}" -c "import vlne, vlndata" 2> /dev/null \
    || die "vlne env not active. Run: conda deactivate; conda activate vlne"

data="${VLNE_DATADIR}/sbnd"
eval_set="${data}/eval_numu.hdf"
train="${data}/train_numu_${tag}.hdf"
shuffled="${data}/train_numu_${tag}_shuffled.hdf"
config="scripts/train/sbnd/numu_${tag}.py"
outdir="${VLNE_OUTDIR}/sbnd/numu_${tag}"

step "1. Eval set ${eval_set}"
if [[ -f "${eval_set}" ]]
then
    echo "exists, skipping"
else
    "${python}" scripts/data/flatcaf_to_hdf5.py "${eval_set}" "${EVAL_DIR}" -j "${JOBS}" \
        || die "Eval set conversion failed"
fi

step "2. Training set ${train}"
if [[ -f "${train}" ]]
then
    echo "exists, skipping (TRAIN_DIRs ignored; delete it to rebuild)"
else
    "${python}" scripts/data/flatcaf_to_hdf5.py "${train}" "${train_dirs[@]}" -j "${JOBS}" \
        || die "Training set conversion failed"
fi

step "3. Checking training files against the eval set"
overlap=$(comm -12 <(sed 's|.*/||' "${train}.files.txt" | sort) \
                   <(sed 's|.*/||' "${eval_set}.files.txt" | sort))
[[ -z "${overlap}" ]] \
    || die "These files are in both the training and eval sets:"$'\n'"${overlap}"
echo "no overlap: $(wc -l < "${train}.files.txt" | tr -d ' ') training files," \
     "$(wc -l < "${eval_set}.files.txt" | tr -d ' ') eval files"

step "4. Shuffled training set ${shuffled}"
if [[ -f "${shuffled}" ]]
then
    echo "exists, skipping"
else
    "${python}" scripts/data/shuffle.py "${train}" "${shuffled}" > /dev/null \
        || die "Shuffle failed"
    echo "done"
fi

step "5. Config ${config}"
if [[ -f "${config}" ]]
then
    echo "exists, skipping"
else
    {
        echo "# Made by run_numu.sh from $(basename "${BASE_CONFIG}");" \
             "inputs listed in ${train}.files.txt"
        sed -e "s|\"sbnd/train_numu[^\"]*\"|\"sbnd/train_numu_${tag}_shuffled.hdf\"|" \
            -e "s|'sbnd/numu_[^']*'|'sbnd/numu_${tag}'|" \
            "${BASE_CONFIG}"
    } > "${config}"

    grep -q "train_numu_${tag}_shuffled.hdf" "${config}" \
        && grep -q "'sbnd/numu_${tag}'" "${config}" \
        || { rm -f "${config}"; die "Could not set the data path/outdir from ${BASE_CONFIG}"; }
    echo "data path sbnd/train_numu_${tag}_shuffled.hdf, outdir sbnd/numu_${tag}"
fi

step "6. Training into ${outdir}"
model=$(ls -td "${outdir}"/model_m* 2> /dev/null | head -n 1)
if [[ -n "${model}" && -f "${model}/model.h5" ]]
then
    echo "trained model exists, skipping (delete ${outdir} to retrain)"
else
    "${python}" "${config}" || die "Training failed"
    model=$(ls -td "${outdir}"/model_m* | head -n 1)
fi

step "7. Eval of ${model}"
"${python}" scripts/eval/eval_model.py --preset uboone_numu --split test --weights none \
    --data sbnd/eval_numu.hdf -e png -- "${model}" \
    || die "Eval failed"

evaldir="${model}/evals/eval_d(eval_numu.hdf)_p(uboone_numu)_s(test)_None"
echo -e "\nRelative resolution (reco - true) / true, network:"
summary "${evaldir}/stats.csv"
echo -e "\nBaseline (reco.nuE, reco.lepE):"
summary "${evaldir}/stats_base.csv"
echo -e "\nPlots: ${evaldir}/plots"
