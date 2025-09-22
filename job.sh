#!/bin/bash
#SBATCH -q preempt
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=20:30:00
#SBATCH -A m4031
#SBATCH -J wl_ml_train
#SBATCH --output=ResNet_baseline.out
#SBATCH --error=ResNet_baseline.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=lindazjin@berkeley.edu

# CPU-only environment settings
export OMP_NUM_THREADS=32
export OMP_PLACES=threads  
export OMP_PROC_BIND=spread
export JAX_PLATFORMS=cpu  # Force JAX to use CPU only
export JAX_ENABLE_X64=true  # Use float32 for better performance

# Load environment
module load python
source /global/u1/l/lindajin/virtualenvs/env1/bin/activate

# Print job information
echo "Job started at: $(date)"
if command -v seff &> /dev/null; then
    echo "Running seff for job $SLURM_JOB_ID..."
    seff $SLURM_JOB_ID
else
    echo "seff command not available"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on nodes: $SLURM_JOB_NODELIST"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
fi
echo "Number of MPI tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OpenMP threads per task: $OMP_NUM_THREADS"

# Change to project directory
cd /pscratch/sd/l/lindajin/WL_ML

# Run CPU-optimized physics-informed neural network training
python -u train_direct.py