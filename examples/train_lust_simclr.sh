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

op="/work/tier3/jkrupa/cl/nov17_simclr_t${t}_fullData/"

echo "Outpath ${op}"
python3 cl_v1_train_t0p1_nloss.py \
	--ipath /work/tier3/jkrupa/cl/samples/nov17/merged_2/train/ \
        --vpath /work/tier3/jkrupa/cl/samples/nov17/merged_2/val/ \
	--opath ${op} \
	--temperature $t \
	--nepochs $n 
