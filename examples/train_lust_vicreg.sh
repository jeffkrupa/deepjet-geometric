#!/bin/bash

while getopts ":v:m:c:n:" o; do
    case "${o}" in
        v)
            v=${OPTARG}
            ;;
        m)
            m=${OPTARG}
            ;;
        c)
            c=${OPTARG}
            ;;
        n)
            n=${OPTARG}
            ;;
    esac
done
shift $((OPTIND-1))
echo "Training VICREG with weights mse=$m, var=$v, cov=$c for $n epochs"

op="/work/tier3/jkrupa/cl/aug16_vicreg_mse${m}_var${v}_cov${c}_fullData/"

echo "Outpath ${op}"
python3 cl_v1_train_vicreg.py \
	--ipath /scratch/submit/cms/bmaier/cl/samples/aug16_train/ \
        --vpath /scratch/submit/cms/bmaier/cl/samples/aug16_val/ \
	--opath ${op} \
	--mse $m \
	--var $v \
	--cov $c \
	--nepochs $n 
