#!/bin/bash

POSITIONAL_ARGS=()
WHICH_AUGMENTATIONS=()
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --nepochs)
      NEPOCHS="$2"
      PYTHON_ARGS+=("--nepochs" "$NEPOCHS")
      shift # past argument
      shift # past value
      ;;
    --ntrain)
      NTRAIN="$2"
      PYTHON_ARGS+=("--Nmaxsample_train" "$NTRAIN")
      shift # past argument
      shift # past value
      ;;
    --fix_weights)
      PYTHON_ARGS+=("--fix_weights")
      shift # past argument
      ;;
    --fully_supervised)
      PYTHON_ARGS+=("--fs_train")
      shift # past argument
      ;;
    --one_layer_MLP)
      PYTHON_ARGS+=("--one_layer_MLP")
      shift # past argument
      ;;
    --mpath)
      MPATH="$2"
      PYTHON_ARGS+=("--mpath" "$MPATH")
      shift # past argument
      shift # past argument
      ;;
    --which_augmentations)
      shift # past argument
          while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
              WHICH_AUGMENTATIONS+=("$1"); shift;
      done
      ;;
    #-*|--*)
    #  echo "Unknown option $1"
    #  exit 1
    #  ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ ${#WHICH_AUGMENTATIONS[@]} -gt 0 ]; then
    PYTHON_ARGS+=("--which_augmentations" "${WHICH_AUGMENTATIONS[@]}")
fi

contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "True"
            return 0
        fi
    }
    echo "False"
    return 1
}


# Call contains function to check for --fix_weights and --fully_supervised in the PYTHON_ARGS
FIX_WEIGHTS=$(contains "${PYTHON_ARGS[@]}" "--fix_weights")
FULLY_SUPERVISED=$(contains "${PYTHON_ARGS[@]}" "--fully_supervised")
ONE_LAYER_MLP=$(contains "${PYTHON_ARGS[@]}" "--one_layer_MLP")
# Echo out the current settings
echo "Current settings:"
echo "NEPOCHS: ${NEPOCHS:-'Not provided'}"
echo "NTRAIN: ${NTRAIN:-'Not provided'}"
echo "FIX_WEIGHTS: ${FIX_WEIGHTS}"
echo "FULLY_SUPERVISED: ${FULLY_SUPERVISED}"
echo "WHICH_AUGMENTATIONS: ${WHICH_AUGMENTATIONS[*]:-'None'}"
echo "MPATH: ${MPATH:-'Not provided'}"
echo "ONE_LAYER_MLP: ${ONE_LAYER_MLP}"

# Make opath unique based on training setup
OLD_IFS="$IFS"
IFS='' opath="nov23/Graph-ntrain=${NTRAIN},augs=${WHICH_AUGMENTATIONS[*]}"
IFS="$OLD_IFS"
if [[ $FULLY_SUPERVISED == true ]]; then
   opath="$opath,fully-supervised"
else
   opath="$opath,fine-tuned"
fi

if [[ $FIX_WEIGHTS == true ]]; then
   opath="$opath,fixed_weights"
else
   opath="$opath,floating_weights"
fi

if [[ $ONE_LAYER_MLP == true ]]; then
   opath="$opath,onelayerMLP"
else
   opath="$opath,fivelayerMLP"
fi

path_without_file="${MPATH%/*}"
extracted_dir="${path_without_file##*/}"
opath="$opath,RS3Lbase=${extracted_dir}"

if [ -d "$opath" ]; then
   echo "path already exists. remove it with:"
   echo "rm -rf ${opath}"
   return 0 2>/dev/null || exit 0
fi
PYTHON_ARGS+=("--opath" "${opath}")

mkdir -p ${opath}
echo python3 cl_v1_train_t0p1_nloss_Nate2.py --ipath /work/tier3/jkrupa/cl/samples/mar20_finetuning/outfiles/train/ --vpath /work/tier3/jkrupa/cl/samples/mar20_finetuning/outfiles/val/ --temperature 0.1 --n_out_nodes 8 --hidden_dim 128 --Nmaxsample_val 1e4 --lr 0.0001 --batchsize 1000 --fine_tuning "${PYTHON_ARGS[@]}" "${POSITIONAL_ARGS[@]}" > ${opath}/runcommand.sh
python3 cl_v1_train_t0p1_nloss_Nate2.py --ipath /work/tier3/jkrupa/cl/samples/mar20_finetuning/outfiles/train/ --vpath /work/tier3/jkrupa/cl/samples/mar20_finetuning/outfiles/val/ --temperature 0.1 --n_out_nodes 8 --hidden_dim 128 --Nmaxsample_val 1e4 --lr 0.0001 --batchsize 1000 --fine_tuning "${PYTHON_ARGS[@]}" "${POSITIONAL_ARGS[@]}" > $opath/output.txt 2>&1
