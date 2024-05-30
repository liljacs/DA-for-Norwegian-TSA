for SEED in {10,20,30,40,50}; do
    slurm_file="run_tsa_$SEED.slurm"

    echo "#!/bin/bash" >> $slurm_file
    echo "Seed: $SEED"
    echo "#SBATCH --job-name=tsa" >> $slurm_file
    echo "#SBATCH --mail-type=FAIL" >> $slurm_file
    echo "#SBATCH --account=ec30" >> $slurm_file
    echo "#SBATCH --time=10:00:00" >> $slurm_file
    echo "#SBATCH --partition=accel" >> $slurm_file
    echo "#SBATCH --gpus=1" >> $slurm_file 
    echo "#SBATCH --nodes=1" >> $slurm_file
    echo "#SBATCH --mem-per-cpu=32G" >> $slurm_file
    echo "#SBATCH --cpus-per-task=6" >> $slurm_file
    echo "#SBATCH --output=/fp/projects01/ec30/liljacs/Master/modeling/slurm_output_large/%j.out" >> $slurm_file

    echo "set -o errexit  # Recommended for easier debugging"

    echo "module purge" >> $slurm_file
    echo "module use -a /fp/projects01/ec30/software/easybuild/modules/all/" >> $slurm_file
    echo "module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8" >> $slurm_file
    echo "module load nlpl-nlptools/01-foss-2022b-Python-3.10.8" >> $slurm_file
    echo "module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8" >> $slurm_file
    echo "module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8" >> $slurm_file
    echo "module load nlpl-datasets/2.15.0-foss-2022b-Python-3.10.8" >> $slurm_file
    echo "module load nlpl-huggingface-hub/0.17.3-foss-2022b-Python-3.10.8" >> $slurm_file
    #echo "pip install wandb" >> $slurm_file # comment out once installed

    # print information (optional)
    echo "submission directory: ${SUBMITDIR}"

    echo "python3 /fp/projects01/ec30/liljacs/Master/modeling/run_model.py --seed ${SEED}" >> $slurm_file
    sbatch $slurm_file
done
