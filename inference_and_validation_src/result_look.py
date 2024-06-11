import json
import fire
import os
OUTPUT_TEMP = """
'======{validated_file} results:===='
Model: {model_type}
PEFT Method: {peft_type}  
train_dataset: {train_dataset}  
validation benchmark : {benchmark_name} 
validation result: 
{validation_result}
"""
def cal_result(
    # validated_file
    validated_file: str = " llmpeft4apr/results_new/CodeLlama_7b_hf_apr_new_lora_on_humaneval_validation_24_04_04_11_46_17.json",
    model_type: str = 'CodeLlama-7b-hf',
    peft_type: str = 'lora',
    train_dataset: str = 'apr_new',
    benchmark_name: str = 'humaneval',

    # output_dir: str = ""
    # pass_k: int = 1
):
    
    validated_result = json.load(open(validated_file, 'r'))
    validated_result = validated_result['data']
    validated_result_fp = validated_file.split('.json')[0] + '_result_look.txt'
    res = []

    pass_k_list = [i for i in range(0,11)]
    for pass_k in pass_k_list:
        total = 0
        plausible_pathces = 0
        for proj in validated_result:
            # if 'output' not in validated_data[proj]:
            #     continue
            # if len(validated_data[proj]['output']) == 0:
            #     continue
            total += 1
            for rank, patch_res in validated_result[proj].items():
                if int(rank) > pass_k - 1 :
                    break
                if patch_res == 'plausible':
                    plausible_pathces += 1
                    # print(proj, rank, patch['patch'])
                    break
        res.append(f'pass@{pass_k}:plausible pathces - {plausible_pathces}, total  problems - {total}, correctness pecent - {plausible_pathces / total}')
    with open(validated_result_fp, 'w') as f:
        f.write(OUTPUT_TEMP.format(validated_file = validated_file, model_type=model_type,peft_type=peft_type,train_dataset=train_dataset,benchmark_name=benchmark_name, validation_result='\n'.join(res)))
if __name__ == '__main__':
    fire.Fire(cal_result)

        