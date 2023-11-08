#!/bin/bash

# Initialize variables
OPATH=""
CONTINUE_TRAINING="--continue_training"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --opath)
      OPATH="$2"
      shift # past argument
      shift # past value
      ;;
    --nepochs)
      NEPOCHS="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Verify that OPATH is provided
if [[ -z "$OPATH" ]]; then
    echo "--opath is required"
    exit 1
fi

# Extract model training details from OPATH
NTRAIN=$(echo $OPATH | grep -oP 'ntrain=\K[0-9e]+')
AUGS=$(echo $OPATH | grep -oP 'augs=\K[\d]+')
FULLY_SUPERVISED=$(echo $OPATH | grep -o 'fully-supervised' &> /dev/null && echo "--fs_train" || echo "")
FIX_WEIGHTS=$(echo $OPATH | grep -o 'fixed_weights' &> /dev/null && echo "--fix_weights" || echo "")
LAYERS=$(echo $OPATH | grep -o 'onelayerMLP' &> /dev/null && echo "--one_layer_MLP")
find_last_model_epoch() {
    local model_dir="$1"
    #echo "Looking in directory: $model_dir"
    local latest_model=$(ls -v "$model_dir"/FT_epoch-*.pt 2>/dev/null | tail -n 1)
    
    echo "$latest_model"
}

# Find the latest model file
LATEST_MODEL=$(find_last_model_epoch "$OPATH")
# Verify that a model file exists
if [[ -z "$LATEST_MODEL" ]]; then
    echo "No model files found in directory: $OPATH"
    return 0 2>/dev/null || exit 0
fi

LATEST_EPOCH=$(basename "$LATEST_MODEL" | grep -oP '(?<=FT_epoch-)\d+(?=\.pt)')
# Check if --nepochs argument is less than the latest model epoch
if [[ -n "$NEPOCHS" && "$NEPOCHS" -le "$LATEST_EPOCH" ]]; then
    echo "Error: The --nepochs argument ($NEPOCHS) is less than or equal to the latest model epoch ($LATEST_EPOCH)."
    echo "Please provide a larger --nepochs value or remove the --continue_training argument to start a new training run."
    return 0 2>/dev/null || exit 0
fi

# Construct the Python command
PYTHON_CMD="python3 cl_v1_train_t0p1_nloss_Nate2.py --ipath /work/tier3/jkrupa/cl/samples/mar20_finetuning/outfiles/train/ --vpath /work/tier3/jkrupa/cl/samples/mar20_finetuning/outfiles/val/ --temperature 0.1 --n_out_nodes 8 --hidden_dim 128 --Nmaxsample_val 2e6 --lr 0.0001 --batchsize 1000 --fine_tuning --nepochs ${NEPOCHS} --Nmaxsample_train ${NTRAIN} --which_augmentations ${AUGS} ${FULLY_SUPERVISED} ${FIX_WEIGHTS} ${LAYERS} ${CONTINUE_TRAINING} --mpath \"${LATEST_MODEL}\""

# Execute the Python command
echo "Running command: $PYTHON_CMD"
#eval $PYTHON_CMD

# End of the script

