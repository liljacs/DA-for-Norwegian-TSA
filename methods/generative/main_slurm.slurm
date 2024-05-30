#!/bin/bash                                                                                                                                                     

#SBATCH --job-name=LILJACSM                                                                                                                                    
#SBATCH --account=ec30                                                                                                                                          
#SBATCH --time=4:00:00  # Max walltime is 150 hours.                                                                                                            
#SBATCH --nodes=1                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                              
#SBATCH --cpus-per-task=6  # 6 CPU cores per task to keep the parallel data feeding going. A little overkill, but CPU time is very cheap compared to GPU time.  
#SBATCH --mem-per-cpu=16G                                                                                                                                       
#SBATCH --partition=accel                                                                                                                                       
#SBATCH --gpus=1   
#SBATCH --output=/fp/projects01/ec30/liljacs/liljacs/slurm_output/%j.out                                                                                                                                  

set -o errexit  # Exit the script on any error                                                                                                                  
set -o nounset  # Treat any unset variables as an error                                                                                                         

# Load modules                                                                                                                                                  
module --quiet purge  # Reset the modules to the system default                                                                                                 
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5

python3 runchat.py $@
