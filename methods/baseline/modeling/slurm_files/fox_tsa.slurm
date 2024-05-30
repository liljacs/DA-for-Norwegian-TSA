#!/bin/bash
#SBATCH --job-name=tsa_master
#SBATCH --account=ec30
#SBATCH --time=23:00:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/fp/projects01/ec30/liljacs/Master/modeling/output/%j.out

#source /fp/projects01/ec30/liljacs/Master/modeling

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-pytorch/1.11.0-foss-2021a-cuda-11.3.1-Python-3.9.5
module load nlpl-datasets/2.3.2-foss-2021a-Python-3.9.5
module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
for SEED in 10 20 30 40 50
do
  echo ${SEED}
  python3 /fp/projects01/ec30/liljacs/Master/modeling/test_model.py ${@}X
done
