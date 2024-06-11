timenow=$(date +\%Y\%m\%d_\%H\%M\%S)
# dir="./"
models_dir="/c21071/lgc/llmpeft4apr/models"
datasets_dir="/c21071/lgc/llmpeft4apr/instruction_tuning_dataset/"
model_type="CodeLlama-7b-hf"
peft_methods="full-ft"
save_dir="/c21071/lgc/llmpeft4apr/codellama_7b_hf/output/$peft_methods/$timenow"

mkdir -p "$save_dir"
# /home/lgc/models/ # 
#prefix:train_batch_size 3, bf True, fp16 False
#p tuning:train_batch_size 4, num_virtual_tokens 20
#debug 
# CUDA_VISIBLE_DEVICES=7 \
# python train_src/sfttrain_peft.py \
# --debug True \
# --model_type  "$model_type" \
# --model_name_or_path "$models_dir/$model_type" \
# --load_in_8bit False \
# --save_dir "$save_dir" \
# --dataset_dir "$datasets_dir" \
# --oss_dataset_path "apr_instruction_total_new.json" \
# --cache_dir "/c21071/lgc/.cache" \
# --max_seq_len 1000 \
# --num_train_epochs 5 \
# --train_batch_size 3 \
# --valid_batch_size 2 \
# --lr 1e-4 \
# --task_type "CAUSAL_LM" \
# --peft_tuning True \
# --peft_methods "$peft_methods" \
# --lora_r $lora_r \
# --lora_alpha $lora_alpha \
# --lora_dropout $lora_dropout \
# --p_tuning_num_virtual_tokens $p_tuning_num_virtual_tokens \
# --p_tuning_encoder_hidden_size $p_tuning_encoder_hidden_size \
# --p_tuning_encoder_reparameterization_type "$p_tuning_encoder_reparameterization_type" \
# --prefix_projection $prefix_projection \
# --prefix_tuning_encoder_hidden_size $prefix_tuning_encoder_hidden_size \
# --prefix_tuning_num_virtual_tokens $prefix_tuning_num_virtual_tokens \
# --gradient_accumulation_steps 1 \
# --use_wandb False \
# --fp16 False \
# --bf16 True \
# --wandb_project "llm4peft" 
# formal
CUDA_VISIBLE_DEVICES=0,1 \
python train_src/sfttrain_ft.py \
--model_type  "$model_type" \
--model_name_or_path "$models_dir/$model_type" \
--load_in_8bit False \
--save_dir "$save_dir" \
--dataset_dir "$datasets_dir" \
--oss_dataset_path "apr_instruction_30k.json" \
--cache_dir "/c21071/lgc/.cache" \
--max_seq_len 1000 \
--num_train_epochs 3 \
--train_batch_size 6 \
--valid_batch_size 2 \
--lr 1e-4 \
--task_type "CAUSAL_LM" \
--peft_methods "$peft_methods" \
--gradient_accumulation_steps 1 \
--use_wandb True \
--fp16 False \
--bf16 True \
--wandb_project "llm4peft" \
> "$save_dir/train.log" 2>&1 &