import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import json
import fire
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
CODELLAMA_INFILLING_TEMPLATE = "▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>"

def create_model_and_tokenizer(model_name_or_path, model_type, load_in_8bit=False):
    """Create model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit,
        device_map="auto",
    )
    # model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path
    )
    # if model_type == 'deepseek-coder-6.7b-base':
    #     tokenizer.pad_token_id = 32018 #"<pad>"
    # else:
    #     tokenizer.pad_token_id = 0 # unk. we want this to be different from the eos token
    # tokenizer.padding_side = "right"  
    # print(model_type + f' pad token id is {tokenizer.pad_token_id}')
    return model, tokenizer

def generate_output(
        #benchmark params
        benchmark_data: str = "",
        benchmark_name: str = "humaneval",
        #output params  
        output_dir: str = "",
        output_file_name: str = "",
        #model params
        model_type: str = "",
        model_name_or_path: str = "/home/survolt/warehouse/llmpeft4apr/models/",
        #generation config params
        num_output: int = 10,
        max_new_tokens: int = 128,
        max_seq_len: int = 1200,
):
    output_data_path = f'{output_dir}{output_file_name}'
    print(f"==========Generating output of {benchmark_name} benchmark by {model_type} ==========")
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    model, tokenizer = create_model_and_tokenizer(model_name_or_path, model_type)    
    codellama_output = json.load(open(benchmark_data, 'r'))
    codellama_output['model'] = model_type
    start_time = time.time()
    for i, proj in enumerate(codellama_output['data']):
        text = codellama_output['data'][proj]['input']

        prefix = text.split("// buggy lines start")[0]
        suffix = text.split("// buggy lines end")[-1]
        # print(f'prefix:\n {prefix}')
        # print(f'suffix:\n {suffix}')
        prompt_text = CODELLAMA_INFILLING_TEMPLATE.format(
                prefix = prefix,
                suffix = suffix
        )
        print(i + 1, 'generating', proj)
        try:
            input_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_tensors="pt",
            )
            eos_id = tokenizer.convert_tokens_to_ids('▁<EOT>')
            generated_ids = model.generate(
                input_ids=input_ids['input_ids'].cuda(), 
                # attention_mask=input_ids['attention_mask'].cuda(),
                max_new_tokens=  len(input_ids[0]) + max_new_tokens, 
                num_beams=num_output, 
                num_return_sequences=num_output, 
                pad_token_id=eos_id, 
                eos_token_id=eos_id,
                early_stopping=True,
            )
            output_list = []

            for generated_id in generated_ids:
                output_list.append(tokenizer.decode(generated_id[len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False))

        except Exception as e:
            output_list = []
            print(e)
        # if model_type == 'deepseek-coder-6.7b-base':
        #     for i, o in enumerate(output_list):
        #         end_bucket = o.rfind('}')
        #         output_list[i] = o[:end_bucket+1]
        # print(output_list[0])
        # break
        for i, o in enumerate(output_list):
            output_list[i] = prefix + '\n' + o + '\n' + suffix
        codellama_output['data'][proj]['output'] = output_list
        json.dump(codellama_output, open(output_data_path, 'w'), indent=2)
    total_time = int(time.time() - start_time)
    codellama_output['time'] = total_time
    codellama_output['benchmark'] = benchmark_name
    json.dump(codellama_output, open(output_data_path, 'w'), indent=2)
    print(f"==========Output written to {output_data_path}==========")
if __name__ == '__main__':
    fire.Fire(generate_output)

    