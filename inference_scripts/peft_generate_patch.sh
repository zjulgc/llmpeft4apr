models_dir="/c21071/lgc/llmpeft4apr/models"
model_type="Llama-2-7b-hf"
output_dir="/c21071/lgc/llmpeft4apr/results/"
peft_methods="p-tuning"
peft_model_weights="/c21071/lgc/llmpeft4apr/llama2_7b_hf/output/p-tuning/20240403_074101/"
train_dataset="apr_new"


CUDA_VISIBLE_DEVICES=4 \
python inference_and_validation_src/peft_generate_patch.py \
--train_dataset "$train_dataset" \
--benchmark_dir "/c21071/lgc/llmpeft4apr/validation_benchmark_dataset/" \
--benchmark_name "humaneval" \
--output_dir "$output_dir" \
--model_type "$model_type" \
--model_name_or_path "$models_dir/$model_type" \
--is_peft True \
--peft_type "$peft_methods" \
--peft_model_weights "$peft_model_weights" \
--num_output 10 \
--max_new_tokens 512 \
--max_seq_len 1024 