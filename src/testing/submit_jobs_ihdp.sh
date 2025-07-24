#!/bin/bash
#SBATCH --job-name=IHDPExperiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --mem=4G
#SBATCH --time=7-0:0:0
#SBATCH --output=log/ihdp_experiment_%A_%a.out
#SBATCH --error=log/error_ihdp_experiment_%A_%a.err
#SBATCH --array=0-1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=housszenati@gmail.com

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load miniconda/4.9.2
source activate gatsby

echo "Running IHDP experiment on CPU core"

# Path to parameter CSV file
INPUT_FILE=/nfs/ghome/live/hzenati/counterfactual-policy-mean-embedding-swc/src/testing/experiment_parameters/ihdp_experiment_parameters.csv

# Read the line corresponding to SLURM_ARRAY_TASK_ID
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $INPUT_FILE)

# Extract fields
scenario=$(echo $LINE | cut -d',' -f2)
method=$(echo $LINE | cut -d',' -f3)
seed=$(echo $LINE | cut -d',' -f4)

# Run the Python experiment
python /nfs/ghome/live/hzenati/counterfactual-policy-mean-embedding-swc/src/testing/experiments_ihdp.py \
    --run \
    --scenario $scenario \
    --method $method \
    --seed $seed

conda deactivate
