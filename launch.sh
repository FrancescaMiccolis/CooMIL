#!/bin/bash
#SBATCH --job-name=conslide
#SBATCH --output=/homes/gcorso/Conslide_dev/outputs/output_conslide.txt
#SBATCH --error=/homes/gcorso/Conslide_dev/errors/error_conslide.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=all_usr_prod
#SBATCH --account=h2020deciderficarra
#SBATCH --time=24:00:00
#SBATCH --mem=64G

touch /homes/gcorso/Conslide_dev/outputs/output_conslide.txt
touch /homes/gcorso/Conslide_dev/errors/error_conslide.txt
cd /homes/gcorso/Conslide_dev/

CUDA_VISIBLE_DEVICES=0 python -u /homes/gcorso/Conslide_dev/utils/main.py --model conslide --dataset seq-wsi --exp_desc conslide --buffer_size 1100 --alpha 0.2 --beta 0.2 --n_epochs 50




