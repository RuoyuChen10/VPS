#!/bin/bash

dataset="datasets/coco/val2017"
eval_list="datasets/coco_florence-2_correct_detection.json"
lambda1=1
lambda2=1
division_number=50
save_dir="./submodular_results/florence-2-coco-detection-correctly/"

declare -a cuda_devices=("0")

# GPU numbers
gpu_numbers=${#cuda_devices[@]}
echo "The number of GPUs is $gpu_numbers."

# text length
line_count_per_gpu=200
echo "Each GPU should process at least $line_count_per_gpu lines."

gpu_index=0
for device in "${cuda_devices[@]}"
do
    begin=$((gpu_index * line_count_per_gpu))
    if [ $gpu_index -eq $((gpu_numbers - 1)) ]; then
        end=-1  # 最后一个 GPU，设置 end 为 -1
    else
        end=$((begin + line_count_per_gpu))
    fi

    CUDA_VISIBLE_DEVICES=$device python -m detection_attribution.florence-2_correct_detection \
    --Datasets $dataset \
    --eval-list $eval_list \
    --lambda1 $lambda1 \
    --lambda2 $lambda2 \
    --division-number $division_number \
    --save-dir $save_dir \
    --begin $begin \
    --end $end &

    gpu_index=$((gpu_index + 1))
done

wait
echo "All processes have completed."