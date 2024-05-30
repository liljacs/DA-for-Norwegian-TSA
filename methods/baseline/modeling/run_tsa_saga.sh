for SEED in {10,20,30,40,50}; do
    slurm_file="run_tsa_$SEED.slurm"

    if [[ -e "${slurm_file}" ]]; then
        rm "${slurm_file}"
    fi

    echo "#!/bin/bash" >> $slurm_file
    echo "Seed: $SEED"
    echo "#SBATCH --job-name=master_tsa" >> $slurm_file
    echo "#SBATCH --mail-type=FAIL" >> $slurm_file
    echo "#SBATCH --account=nn9851k" >> $slurm_file
    echo "#SBATCH --time=5:00:00" >> $slurm_file
    echo "#SBATCH --partition=a100" >> $slurm_file
    echo "#SBATCH --gpus=1" >> $slurm_file 
    echo "#SBATCH --nodes=1" >> $slurm_file
    echo "#SBATCH --mem-per-cpu=32G" >> $slurm_file
    echo "#SBATCH --cpus-per-task=6" >> $slurm_file
    echo "#SBATCH --output=output_prompt/slurm_prompt_1/%j.out" >> $slurm_file

    echo "set -o errexit  # Recommended for easier debugging"

    #echo "module purge" >> $slurm_file
    echo "module --force swap StdEnv Zen2Env" >> $slurm_file
    echo "module use -a /cluster/shared/nlpl/software/eb/etc/all/" >> $slurm_file
    echo "module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8.lua" >> $slurm_file
    echo "module load nlpl-nlptools/01-foss-2022b-Python-3.10.8.lua" >> $slurm_file
    echo "module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8.lua" >> $slurm_file
    echo "module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8.lua" >> $slurm_file
    echo "module load nlpl-datasets/2.15.0-foss-2022b-Python-3.10.8.lua" >> $slurm_file
    echo "module load nlpl-huggingface-hub/0.17.3-foss-2022b-Python-3.10.8.lua" >> $slurm_file
    #echo "pip install wandb" >> $slurm_file # comment out once installed

    # print information (optional)
    echo "submission directory: ${SUBMITDIR}"

    echo "python3 /cluster/home/liljacs/Master/modeling/run_model.py --seed ${SEED}" >> $slurm_file
    sbatch $slurm_file
done
