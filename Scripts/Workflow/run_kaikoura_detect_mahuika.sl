#!/bin/bash
#SBATCH -J Kaik_aft_gaps
#SBATCH -A nesi02337
#SBATCH --time=1:30:00
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --output=/nesi/project/nesi02337/kaikoura-afterslp/Logs/detect_out_%a.txt
#SBATCH --error=/nesi/project/nesi02337/kaikoura-afterslp/Logs/detect_err_%a.txt
#SBATCH --cpus-per-task=36
#SBATCH --array=151,155,156,159,160,163,165,168,170,259
# full run: #SBATCH --array=88-577

module load Python/3.6.3-gimkl-2017a

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE

# Use for main run (without profiling)
srun python /nesi/project/nesi02337/kaikoura-afterslp/Scripts/kaikoura_detect.py -d $SLURM_ARRAY_TASK_ID -s 1

