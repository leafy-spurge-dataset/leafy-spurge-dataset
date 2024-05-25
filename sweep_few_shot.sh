#!/bin/bash

cd ~/leafy-spurge-dataset

for RANDOM_SEED in 0 1 2 3 4; do 
for EXAMPLES_PER_CLASS in 1 2 4 8 16 32 64 128 256; do

EXPORT_VARS=RANDOM_SEED=$RANDOM_SEED
EXPORT_VARS=$EXPORT_VARS,EXAMPLES_PER_CLASS=$EXAMPLES_PER_CLASS

sbatch --export=$EXPORT_VARS few_shot.sbatch

done
done