#!/bin/bash
#SBATCH --job-name=master_tsa
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --time=5:00:00
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=output_prompt/slurm_prompt_1/%j.out
module --force swap StdEnv Zen2Env
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8.lua
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8.lua
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8.lua
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8.lua
module load nlpl-datasets/2.15.0-foss-2022b-Python-3.10.8.lua
module load nlpl-huggingface-hub/0.17.3-foss-2022b-Python-3.10.8.lua
python3 /cluster/home/liljacs/Master/modeling/run_model.py --seed 10
