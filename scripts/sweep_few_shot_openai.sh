#!/bin/bash

cd ~/leafy-spurge-dataset

for DATASET_VERSION in crop context; do 
for MODEL in gpt-4o gpt-4-turbo; do 
for RANDOM_SEED in 0 1 2 3 4 5 6 7; do 
for EXAMPLES_PER_CLASS in 1 2 4 8 16; do

EXPORT_VARS=DATASET_VERSION=$DATASET_VERSION
EXPORT_VARS=$EXPORT_VARS,MODEL=$MODEL
EXPORT_VARS=$EXPORT_VARS,RANDOM_SEED=$RANDOM_SEED
EXPORT_VARS=$EXPORT_VARS,EXAMPLES_PER_CLASS=$EXAMPLES_PER_CLASS

sbatch --export=$EXPORT_VARS few_shot_openai.sbatch

done
done
done
done