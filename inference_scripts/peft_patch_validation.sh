python inference_and_validation_src/peft_patch_validation.py \
--input_file  " llmpeft4apr/results/deepseek_coder_6.7b_base_code_alpaca_lora_on_humaneval_output_24_03_30_10_38_22.json" \
--output_dir " llmpeft4apr/results/" \
--benchmark_dir " llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java/" \
--benchmark_name "humaneval" \
--peft_type "lora" \
--model_type "deepseek-coder-6.7b-base" \
--train_dataset "alpaca"
