#!/bin/bash
#SBATCH --job-name=tsa_master_baseline
#SBATCH --account=NN9851K
#SBATCH --time=23:00:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=32G
#SBATCH --output=output/%j.out

#source $/cluster/home/liljacs/Master/modeling

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module --quiet purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-python-candy/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-transformers/4.24.0-gomkl-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
for SEED in 10 20 30 40 50
do
  python3 /cluster/home/liljacs/Master/modeling/test_model.py -s ${SEED}
  echo ${SEED}
done
