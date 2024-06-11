import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import fire
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel,PeftConfig
from prompter import Prompter
from datetime import datetime
import re
# from humaneval_patch_validate import validate_humaneval
# from quixbugs_patch_validate import validate_quixbugs
# from  defects4j_patch_validate import validate_defects4j
# from mbjp_apr_patch_validate import validate_mbjp_apr
# from result_look import cal_result
BENCHMARK_PROMPT = """Write a solution to the following coding problem:
The input is buggy code, which bug lines start from '// buggy lines start' and end at '// buggy lines end'. Please fix the follwing code. 
{problem}"""
def create_model_and_tokenizer(model_name_or_path, model_type, ft_model_weights, load_in_8bit=False):
    """Create model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ft_model_weights,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        device_map="auto",
    )
    # model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path
    )
    if model_type == 'deepseek-coder-6.7b-base':
        tokenizer.pad_token_id = 32018 #"<pad>"
    else:
        tokenizer.pad_token_id = 0 # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"  
    print(model_type + f' pad token id is {tokenizer.pad_token_id}')
    return model, tokenizer

def oss_response_filter(output):
    pattern = r'```java(.*?)```'
    matches = re.findall(pattern, output, re.DOTALL)
    if len(matches) == 1:
        liens = matches[0].split('\n')
        for l in liens:
            if '@@' in l or '@Override' in l:
                liens.remove(l)
            
        return '\n'.join(liens)
    else:
        return ""
    
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
        train_dataset: str = "magicoder",
        #peft model params
        ft_model_weights: str = "/home/survolt/warehouse/llmpeft4apr/models/", 
        #generation config params
        num_output: int = 10,
        max_new_tokens: int = 256,
        max_seq_len: int = 1200,

):
    #format input and output path
    benchmark_data_path = benchmark_data
    # print(benchmark_data_path)
    output_data_path = f'{output_dir}{output_file_name}'
    print(f"==========Generating output of {benchmark_name} benchmark by {model_type} ==========")
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # create base-model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_name_or_path, model_type, ft_model_weights)
    model.eval()
    #create prompter
    prompter = Prompter()
    #merge peft model weights
    output = json.load(open(benchmark_data_path, 'r'))
    output['model'] = model_type
    output['train_dataset'] = train_dataset
    start_time = time.time()
    for i, proj in enumerate(output['data']):
        text = output['data'][proj]['input']

        prompt_text = prompter.generate_prompt(BENCHMARK_PROMPT.format(problem=text))

        print(i + 1, 'generating', proj)

        try:
            input_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_tensors="pt",
            )
            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids['input_ids'].cuda(), 
                    # attention_mask=input_ids['attention_mask'].cuda(),
                    max_new_tokens=  len(input_ids[0]) + max_new_tokens, 
                    num_beams=num_output, 
                    num_return_sequences=num_output, 
                    pad_token_id=tokenizer.pad_token_id, 
                    eos_token_id=eos_id,
                )
            output_list = []

            for generated_id in generated_ids:
                output_list.append(tokenizer.decode(generated_id[len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                # print(output_list[-1])
                # print('==============')

        except Exception as e:
            output_list = []
            print(e)
        # if train_dataset == 'magicoder':
        #     for i, o in enumerate(output_list):
        #         output_list[i] = oss_response_filter(o)
        # if model_type == 'deepseek-coder-6.7b-base':
        #     for i, o in enumerate(output_list):
        #         end_bucket = o.rfind('}')
        #         output_list[i] = o[:end_bucket+1]
        output['data'][proj]['output'] = output_list

        json.dump(output, open(output_data_path, 'w'), indent=2)
        # break
    total_time = int(time.time() - start_time)
    output['time'] = total_time
    output['benchmark'] = benchmark_name
    json.dump(output, open(output_data_path, 'w'), indent=2)
    print(f"==========Output written to {output_data_path}==========")

if __name__ == '__main__':
    fire.Fire(generate_output)

    