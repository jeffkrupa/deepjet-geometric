#!/bin/bash

while getopts ":t:n:" o; do
    case "${o}" in
        t)
            t=${OPTARG}
            ;;
        n)
            n=${OPTARG}
            ;;
    esac
done
shift $((OPTIND-1))
echo "Training SIMCLR with temperature t=$t, for $n epochs"

op="/work/tier3/jkrupa/cl/aug16_simclr_t${t}_fullData/"

echo "Outpath ${op}"
python3 cl_v1_train_t0p1_nloss.py \
	--ipath /scratch/submit/cms/bmaier/cl/samples/aug16_train/ \
        --vpath /scratch/submit/cms/bmaier/cl/samples/aug16_val/ \
	--opath ${op} \
	--temperature $t \
	--nepochs $n 
