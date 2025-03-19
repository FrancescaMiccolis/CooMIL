#!/bin/bash
#SBATCH --job-name=coomil
#SBATCH --output=/homes/gcorso/outputs/output_coomil.txt
#SBATCH --error=/homes/gcorso/errors/error_coomil.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=all_serial
#SBATCH --account=h2020deciderficarra
#SBATCH --time=04:00:00

touch /homes/gcorso/outputs/output_coomil.txt
touch /homes/gcorso/errors/error_coomil.txt
cd /homes/gcorso/Conslide_dev/

base="
--model, conslide, --dataset, seq-wsi, --exp_desc, conslide, --buffer_size, 1100, --alpha, 0.2, --beta, 0.2
"

srun -N1 -n1 --gres=gpu:1 python -u -m /homes/gcorso/Conslide_dev/utils/main ${base} &

wait




