#!/bin/bash
#SBATCH --job-name=leafy-spurge
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-1-8,matrix-1-18,matrix-1-20,matrix-0-34,matrix-1-16,matrix-0-38,matrix-1-6,matrix-2-5,matrix-1-10,matrix-2-21,matrix-2-1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --output=/home/btrabucc/leafy-spurge-dataset/logs/log-%A-%a-%N.out
#SBATCH --error=/home/btrabucc/leafy-spurge-dataset/logs/err-%A-%a-%N.out
#SBATCH --mail-type=END
#SBATCH --mail-user=btrabucc@andrew.cmu.edu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate leafy-spurge
cd ~/leafy-spurge-dataset

if [ -z "${EXAMPLES_PER_CLASS+xxx}" ]; then echo "EXAMPLES_PER_CLASS is not set" && exit 1; fi
if [ -z "${RANDOM_SEED+xxx}" ]; then echo "RANDOM_SEED is not set" && exit 1; fi
if [ -z "${MODEL+xxx}" ]; then echo "MODEL is not set" && exit 1; fi
if [ -z "${DATASET_VERSION+xxx}" ]; then echo "DATASET_VERSION is not set" && exit 1; fi

OUTPUT_DIR="/projects/rsalakhugroup/btrabucc/leafy-spurge-results/version=$DATASET_VERSION-model=$MODEL-n=$EXAMPLES_PER_CLASS-seed=$RANDOM_SEED"

accelerate launch -u -m leafy_spurge_dataset.experiments.train_classifier \
    --dataset_version $DATASET_VERSION \
    --model $MODEL \
    --examples_per_class $EXAMPLES_PER_CLASS \
    --batch_size 32 \
    --num_epochs 50 \
    --steps_per_epoch 100 \
    --lr 0.0001 \
    --seed $RANDOM_SEED \
    --output_dir $OUTPUT_DIR