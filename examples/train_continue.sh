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
      return 0 2>/dev/null || exit 0
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
NVAL=$(echo $OPATH | grep -oP 'nval=\K[0-9e]+')
AUGS=$(echo $OPATH | grep -oP 'augs=\K[\d]+')
FULLY_SUPERVISED=$(echo $OPATH | grep -o 'fully-supervised' &> /dev/null && echo "--fs_train" || echo "")
FIX_WEIGHTS=$(echo $OPATH | grep -o 'fixed_weights' &> /dev/null && echo "--fix_weights" || echo "")
LAYERS=$(echo $OPATH | grep -o 'onelayerMLP' &> /dev/null && echo "--one_layer_MLP")
WZ_ZZ=$(echo $OPATH | grep -o 'wz_zz' &> /dev/null && echo "--wz_zz")
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

if [[ "$(hostname)" == *"satori"* ]]; then
   basepath="/nobackup/users/bmaier/rs3l/"
else
   basepath="/work/tier3/jkrupa/cl/samples/"
fi

if [[ "$WZ_ZZ" == "True" ]]; then
   opath="$opath,wz_zz"
   ipath="$basepath/mar20/wz-vs-zz/train/"
   vpath="$basepath/mar20/wz-vs-zz/val/"
else
   opath="$opath,h_qcd"
   ipath="$basepath/mar20_finetuning/outfiles/train/"
   vpath="$basepath/mar20_finetuning/outfiles/val/"
fi


# Construct the Python command
PYTHON_CMD="python3 cl_v1_train_t0p1_nloss_Nate2.py --ipath ${ipath} --vpath ${vpath} --temperature 0.1 --n_out_nodes 8 --hidden_dim 128 --Nmaxsample_val 2e6 --lr 0.0001 --batchsize 1000 --fine_tuning --nepochs ${NEPOCHS} --Nmaxsample_train ${NTRAIN} --which_augmentations "$(echo "$AUGS" | sed 's/./& /g' | sed 's/ $//')" ${FULLY_SUPERVISED} ${FIX_WEIGHTS} ${LAYERS} ${CONTINUE_TRAINING} ${WZ_ZZ} --opath ${OPATH} --mpath \"${LATEST_MODEL}\"" 

# Execute the Python command
echo "Running command:" >>  ${OPATH}/output.txt
echo "$PYTHON_CMD" >> ${OPATH}/output.txt
echo "$PYTHON_CMD" > ${OPATH}/runcommand_continue.sh
#eval $PYTHON_CMD >> ${OPATH}/output.txt

# End of the script



#echo python3 cl_v1_train_t0p1_nloss_Nate2.py --ipath ${ipath} --vpath ${vpath} --temperature 0.1 --n_out_nodes 8 --hidden_dim 128 --lr 0.0001 --batchsize 1000 --fine_tuning --continue_training "${PYTHON_ARGS[@]}" "${POSITIONAL_ARGS[@]}" > ${opath}/runcommand_continue.sh

opath="$OPATH"
rm ${opath}/sub.sh
echo "#!/bin/bash " >> ${opath}/sub.sh

echo "#SBATCH -J rsXX3l_1GPU"  >> ${opath}/sub.sh
echo "#SBATCH -o logs/rs3l_1GPUs_%j.out ">> ${opath}/sub.sh
echo "#SBATCH -e logs/rs3l_1GPUs_%j.err ">> ${opath}/sub.sh
#echo "#SBATCH --mail-user=jkrupa@mit.edu ">> ${opath}/sub.sh
echo "#SBATCH --mail-type=ALL ">> ${opath}/sub.sh
if [[ "$(hostname)" == *"satori"* ]]; then
    #echo "#SBATCH --partition=sched_system_all_8" >> ${opath}/sub.sh
    echo "#SBATCH --qos=sched_level_2" >> ${opath}/sub.sh
    echo "#SBATCH --time=24:00:00 ">> ${opath}/sub.sh
    echo "#SBATCH --mem=100G" >> ${opath}/sub.sh
    echo "#SBATCH --gres=gpu:1" >> ${opath}/sub.sh
    #echo "#SBATCH --gpus-per-node=1" >> ${opath}/sub.sh
    echo "#SBATCH --nodes=1" >> ${opath}/sub.sh
else
    echo "#SBATCH --gres=gpu:1 " >> ${opath}/sub.sh
    echo "#SBATCH --partition=submit-gpu ">> ${opath}/sub.sh
    echo "#SBATCH --time=124:00:00 ">> ${opath}/sub.sh
fi 
#

## Number of total processes
#echo " "
echo " echo Nodelist:= " $SLURM_JOB_NODELIST >> ${opath}/sub.sh
echo " echo Number of nodes:= " $SLURM_JOB_NUM_NODES >> ${opath}/sub.sh
echo " echo GPUs per node:= " $SLURM_JOB_GPUS >> ${opath}/sub.sh
echo " echo Ntasks per node:= "  $SLURM_NTASKS_PER_NODE >> ${opath}/sub.sh

if [[ "$(hostname)" == *"satori"* ]]; then
    ## User python environment
    echo 'HOME2=/nobackup/users/bmaier/rs3l/' >> ${opath}/sub.sh
    echo 'PYTHON_VIRTUAL_ENVIRONMENT=rs3l38' >> ${opath}/sub.sh
    echo 'CONDA_ROOT=$HOME2/anaconda3' >> ${opath}/sub.sh

    ## Activate WMLCE virtual environment
    echo 'source ${CONDA_ROOT}/etc/profile.d/conda.sh ' >> ${opath}/sub.sh
    echo 'conda activate $PYTHON_VIRTUAL_ENVIRONMENT' >> ${opath}/sub.sh

    echo "cd /home/$(whoami)/rs3l/deepjet-geometric/" >> ${opath}/sub.sh
    echo 'export PYTHONPATH=${PYTHONPATH}:${PWD}' >> ${opath}/sub.sh
    echo "cd /home/$(whoami)/rs3l/deepjet-geometric/examples/" >> ${opath}/sub.sh
    echo "$(cat ${opath}/runcommand_continue.sh) > ${opath}/output.txt" >> ${opath}/sub.sh
else
    echo "singularity exec --nv --env PYTHONPATH=\"/work/tier3/jkrupa/cl/deepjet-geometric/\" --bind /work/tier3/nswood/cl/ --bind /work/tier3/jkrupa/cl /work/tier3/bmaier/sandboxes/geometricdl.sif $(cat ${opath}/runcommand_continue.sh) >> ${opath}/output.txt" >> ${opath}/sub.sh

fi


sbatch ${opath}/sub.sh
