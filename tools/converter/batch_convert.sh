#!/bin/bash

set -e

WORKDIR='.'

MODEL_TO_CONVERT=(
)

PATH_TO_SAVE=(
)

for i in "${!MODEL_TO_CONVERT[@]}"; do
    path_model=${MODEL_TO_CONVERT[i]}
    path_save=${PATH_TO_SAVE[i]}

    echo $i ":" $path_model "to" $path_save

    python ${WORKDIR}/convert/convert_gpt_to_transformers.py \
        --load_path ${path_model} \
        --save_path ${path_save} \
        --max_shard_size "10GB" \
        --tokenizer_name "decapoda-research/llama-7b-hf" \
        --print-checkpoint-structure
done
