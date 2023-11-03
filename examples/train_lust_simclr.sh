#!/bin/bash

while getopts ":t:n:i:o:p:" x; do
    case "${x}" in
        t)
            t=${OPTARG}
            ;;
        n)
            n=${OPTARG}
            ;;
	i)
	    i=${OPTARG}
	    ;;
	o)
	    o=${OPTARG}
	    ;;
	p)
	    p=${OPTARG}
	    ;;
    esac
done
echo ${output}
shift $((OPTIND-1))
echo "Training SIMCLR with temperature t=$t, for $n epochs"
echo ${output}
op="/work/tier3/jkrupa/cl/simclr_t${t}_d${o}_${p}/"

echo "Outpath ${op}"
python3 cl_v1_train_t0p1_nloss.py \
	--ipath ${i}/train/ \
        --vpath ${i}/val/ \
	--opath ${op} \
	--temperature ${t} \
	--nepochs ${n} \
	--n_out_nodes ${o}
