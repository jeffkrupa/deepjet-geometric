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
    --nval)
      NVAL="$2"
      PYTHON_ARGS+=("--Nmaxsample_val" "$NVAL")
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
    --wz_zz)
      PYTHON_ARGS+=("--wz_zz")
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      return 0 2>/dev/null || exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ ${#WHICH_AUGMENTATIONS[@]} -gt 0 ]; then
    PYTHON_ARGS+=("--which_augmentations" "${WHICH_AUGMENTATIONS[@]}")
fi

# Function to increment the suffix for the directory name
increment_suffix() {
    local path=$1
    local base=$2
    local suffix=1
    if [[ -d "$path" ]]; then
        while [[ -d "${path},repeatedrun${suffix}" ]]; do
            ((suffix++))
        done
        path="${base},repeatedrun${suffix}"
    fi
    echo "$path"
}

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
FULLY_SUPERVISED=$(contains "${PYTHON_ARGS[@]}" "--fs_train")
ONE_LAYER_MLP=$(contains "${PYTHON_ARGS[@]}" "--one_layer_MLP")
WZ_ZZ=$(contains "${PYTHON_ARGS[@]}" "--wz_zz")
# Echo out the current settings
echo "Current settings:"
echo "NEPOCHS: ${NEPOCHS:-'Not provided'}"
echo "NTRAIN: ${NTRAIN:-'Not provided'}"
echo "NVAL: ${NVAL:-'Not provided'}"
echo "FIX_WEIGHTS: ${FIX_WEIGHTS}"
echo "FULLY_SUPERVISED: ${FULLY_SUPERVISED}"
echo "WHICH_AUGMENTATIONS: ${WHICH_AUGMENTATIONS[*]:-'None'}"
echo "MPATH: ${MPATH:-'Not provided'}"
echo "ONE_LAYER_MLP: ${ONE_LAYER_MLP}"
echo "WZ_ZZ: ${WZ_ZZ}"

# Make opath unique based on training setup
OLD_IFS="$IFS"
IFS='' opath="nov23/Graph-ntrain=${NTRAIN},nval=${NVAL},augs=${WHICH_AUGMENTATIONS[*]}"
IFS="$OLD_IFS"
if [[ "$FULLY_SUPERVISED" == "True" ]]; then
   opath="$opath,fully-supervised"
else
   opath="$opath,fine-tuned"
fi

if [[ $FIX_WEIGHTS == "True" ]]; then
   opath="$opath,fixed_weights"
else
   opath="$opath,floating_weights"
fi

if [[ "$ONE_LAYER_MLP" == "True" ]]; then
   opath="$opath,onelayerMLP"
else
   opath="$opath,fivelayerMLP"
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

path_without_file="${MPATH%/*}"
extracted_dir="${path_without_file##*/}"
if [[ "$FULLY_SUPERVISED" == "False" ]]; then 
   opath="$opath,RS3Lbase=${extracted_dir}"
fi

opath=$(increment_suffix "$opath" "$opath")
echo "OPATH: ${opath}"

#if [ -d "$opath" ]; then
#   echo "path already exists. remove it with:"
#   echo "rm -rf ${opath}"
#   return 0 2>/dev/null || exit 0
#fi
PYTHON_ARGS+=("--opath" "${opath}")

mkdir -p ${opath}
echo "New directory created: ${opath}"

#cd /work/tier3/jkrupa/cl/deepjet-geometric/examples

cp cl_v1_train_t0p1_nloss_Nate2.py ${opath}


echo python3 cl_v1_train_t0p1_nloss_Nate2.py --ipath ${ipath} --vpath ${vpath} --temperature 0.1 --n_out_nodes 8 --hidden_dim 128 --lr 0.0001 --batchsize 1000 --fine_tuning "${PYTHON_ARGS[@]}" "${POSITIONAL_ARGS[@]}" > ${opath}/runcommand.sh


rm ${opath}/sub.sh
echo "#!/bin/bash " >> ${opath}/sub.sh

echo "#SBATCH -J rsXX3l_1GPU"  >> ${opath}/sub.sh
echo "#SBATCH -o rs3l_1GPUs_%j.out ">> ${opath}/sub.sh
echo "#SBATCH -e rs3l_1GPUs_%j.err ">> ${opath}/sub.sh
echo "#SBATCH --mail-user=jkrupa@mit.edu ">> ${opath}/sub.sh
echo "#SBATCH --mail-type=ALL ">> ${opath}/sub.sh
if [[ "$(hostname)" == *"satori"* ]]; then
    #echo "#SBATCH --qos=sched_level_2" >> ${opath}/sub.sh
    #echo "#SBATCH --partition=sched_system_all_8" >> ${opath}/sub.sh
    echo "#SBATCH --time=24:00:00 ">> ${opath}/sub.sh
    echo "#SBATCH --mem=100G" >> ${opath}/sub.sh
    echo "#SBATCH --gres=gpu:1" >> ${opath}/sub.sh
    #echo "#SBATCH --gpus-per-node=1" >> ${opath}/sub.sh
    echo "#SBATCH --nodes=1" >> ${opath}/sub.sh
else
    echo "#SBATCH --partition=submit-gpu ">> ${opath}/sub.sh
    echo "#SBATCH --time=96:00:00 ">> ${opath}/sub.sh
fi 
#

## Number of total processes
echo " "
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
    echo "$(cat ${opath}/runcommand.sh) > ${opath}/output.txt" >> ${opath}/sub.sh
else
    echo "singularity exec --nv --env PYTHONPATH=\"/work/tier3/jkrupa/cl/deepjet-geometric/\" --bind /work/tier3/nswood/cl/ --bind /work/tier3/jkrupa/cl /work/tier3/bmaier/sandboxes/geometricdl.sif $(cat ${opath}/runcommand.sh) > ${opath}/output.txt" >> ${opath}/sub.sh
fi
   
#echo "singularity exec --nv --env PYTHONPATH="/work/tier3/jkrupa/cl/deepjet-geometric/" --bind /work/tier3/nswood/cl/ --bind /work/tier3/jkrupa/cl /work/tier3/bmaier/sandboxes/geometricdl.sif "python3 $(echo ${opath}/runcommand.sh)  > $opath/output.txt"" >> ${opath}/sub.sh

echo "echo "Run completed at:- "" >>${opath}/sub.sh
echo date >> ${opath}/sub.sh

#sbatch ${opath}/sub.sh

#curl  -X POST -H 'Content-type: application/json' \
# --data '{"text":"Training Job Completed on '"$(hostname)"', tmux-session '"$(tmux display-message -p '#S')"'."}' \
# https://hooks.slack.com/services/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

