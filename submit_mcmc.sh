#!/bin/bash
#SBATCH -c 200 # Number of cores
#SBATCH -t 0-08:00 # Time
##SBATCH -p arguelles_delgado #Partition, but this line is commented out
#SBATCH -p shared # Partition
#SBATCH --mem=20G # Memory
#SBATCH -o cosmo_final_mcmc.out # Output file
#SBATCH -e cosmo_final_mcmc.err # Error file

source source /n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/setup.sh
date
python cosmo_mcmc_production.py $SLURM_ARRAY_TASK_ID
date