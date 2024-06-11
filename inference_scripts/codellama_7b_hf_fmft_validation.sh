#!/bin/bash
timenow=$(date +\%Y\%m\%d_\%H\%M\%S)
# timenow='20240415_071242'
models_dir=" llmpeft4apr/models"
output_dir=" llmpeft4apr/results/"
# to modify
model_type="CodeLlama-7b-hf"
ft_model_weights=' llmpeft4apr/codellama_7b_hf/output/full-ft/20240513_054604/'
peft_methods='fmft'
train_dataset='apr'

benchmark_names=("humaneval" "quixbugs" "defects4j")
for benchmark_name in "${benchmark_names[@]}"; do
    output_file_name=$model_type'_'$peft_methods'_on_'$benchmark_name'_output_'$timenow'.json'
    #to modify
    CUDA_VISIBLE_DEVICES=7 \
    python inference_and_validation_src/fmft_generate_patch.py \
    --output_file_name $output_file_name \
    --benchmark_data " llmpeft4apr/validation_benchmark_dataset/$benchmark_name.json" \
    --benchmark_name $benchmark_name \
    --output_dir "$output_dir" \
    --model_type "$model_type" \
    --model_name_or_path "$models_dir/$model_type" \
    --ft_model_weights $ft_model_weights \
    --num_output 10 \
    --max_new_tokens 256 \
    --max_seq_len 1024 
done
# benchmark_names=( "quixbugs" "humaneval" )
base_tmp_dir=' llmpeft4apr/tmp_benchmark'
echo "Start validation..."
for benchmark_name in "${benchmark_names[@]}"; do
    output_file_name=$model_type'_'$peft_methods'_on_'$benchmark_name'_output_'$timenow'.json'
    if [ "$benchmark_name" = "humaneval" ]; then
        benchmark_dir=" llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java/"
        tmp_dir=$benchmark_name'_'$timenow
        cp -r $benchmark_dir $base_tmp_dir'/'$tmp_dir'/'
        python inference_and_validation_src/peft_patch_validation.py \
        --input_file  $output_dir$output_file_name  \
        --output_dir $output_dir \
        --benchmark_dir $base_tmp_dir'/'$tmp_dir'/' \
        --benchmark_name $benchmark_name \
        --peft_type $peft_methods \
        --model_type $model_type \
        --train_dataset $train_dataset
        rm -rf $base_tmp_dir'/'$tmp_dir'/'
    elif [ "$benchmark_name" = "quixbugs" ]; then
        benchmark_dir=' llmpeft4apr/validation_benchmark_dataset/benchmarks/quixbugs/'
        tmp_dir=$benchmark_name'_'$timenow
        cp -r $benchmark_dir $base_tmp_dir'/'$tmp_dir
        python inference_and_validation_src/peft_patch_validation.py \
        --input_file  $output_dir$output_file_name  \
        --output_dir $output_dir \
        --benchmark_dir $base_tmp_dir'/'$tmp_dir \
        --benchmark_name $benchmark_name \
        --peft_type $peft_methods \
        --model_type $model_type \
        --train_dataset $train_dataset
        rm -rf $base_tmp_dir'/'$tmp_dir
    elif [ "$benchmark_name" = "defects4j" ]; then
        tmp_dir=$benchmark_name'_'$timenow
        python inference_and_validation_src/peft_patch_validation.py \
        --input_file  $output_dir$output_file_name  \
        --output_dir $output_dir \
        --benchmark_dir $base_tmp_dir'/'$tmp_dir'/' \
        --benchmark_name $benchmark_name \
        --peft_type $peft_methods \
        --model_type $model_type \
        --train_dataset $train_dataset \
        --validation_time '_'$timenow 

        rm -rf $base_tmp_dir'/'$tmp_dir'/'
    else
        echo "Wrong benchmark name!"
    fi
done