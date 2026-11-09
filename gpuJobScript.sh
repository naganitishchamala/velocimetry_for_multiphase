#!/bin/bash
# 
#SBATCH -t 0-01:00:00 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=cavitation
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --mail-user=naga@vt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=sin48

# Loading required modules
module purge
module reset
module load GCC/13.3.0 Python/3.12.3-GCCcore-13.3.0 FFmpeg/7.0.2-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate environment
source "$HOME/workEnv/bin/activate"

# ===============================================
# ALGORITHM EXECUTION EXAMPLES
# ===============================================
# Uncomment the algorithm you wish to run below.
# Make sure to replace the placeholder paths with your actual video directory!

# 1. RAFT (Deep Learning Optical Flow)
# Highly accurate, leverages the requested A30 GPU automatically. Requires an optical flow .pth model weight.
python3 tracker.py --method raft --model weights/raft-sintel.pth --path /scratch/naga/48.cine --use_clahe --throat_loc 331 85 --imgScale 0.04917372 --fpsCam 130000

# 2. OpenPIV (Classical Particle Image Velocimetry)
# Operates strictly on the CPU. Automatically allocates worker pools matching your SLURM --cpus-per-task limit.
# Note: When running OpenPIV, you must uncomment the SBATCH command below to request CPU cores from ARC.
#python3 tracker.py --method openpiv --path /scratch/naga/48.cine --use_clahe --throat_loc 331 85 --imgScale 0.04917372 --fpsCam 130000

# 3. Farneback (Classical Optical Flow)
# Fast, baseline optical flow based on OpenCV's internal implementations.
#python3 tracker.py --method farneback --path /scratch/naga/48.cine --use_clahe --throat_loc 331 85 --imgScale 0.04917372 --fpsCam 130000
