import json
import os
import shutil
import time
import subprocess
import fire
from datetime import datetime
def exec_command_with_timeout(cmd, timeout=60):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    out, err = p.communicate()
    return out, err

def clean_and_create_benchmark_dir(benchmark_dir, benchmark_name):
    #clean buggy and test workspace
    exec_command_with_timeout(['rm', '-rf', benchmark_dir + f'src/main/java/{benchmark_name}/buggy/'])
    
    exec_command_with_timeout(['rm', '-rf', benchmark_dir + f'src/test/java/{benchmark_name}/'])
    #create new workspace
    exec_command_with_timeout(['mkdir', benchmark_dir + f'src/main/java/{benchmark_name}/buggy/'])
    exec_command_with_timeout(['mkdir', benchmark_dir + f'src/test/java/{benchmark_name}/'])
def recopy_proj_code_and_test_case(benchmark_dir, proj, benchmark_name):
    #copy code from src_back

    shutil.copyfile(benchmark_dir + f'src_bak/test/java/{benchmark_name}/TEST_' + proj + '.java', benchmark_dir + f'src/test/java/{benchmark_name}/TEST_' + proj + '.java')

def validate_humaneval(
    #patches generated, json file
    #make sure format of  ouput patch
    input_file: str = " llmpeft4apr/results/deepseek_coder_6.7b_base_apr_p-tuning_on_humaneval_output_24_03_31_13_27_27.json", 
    output_dir: str = " llmpeft4apr/results/", #validation results
    benchmark_dir: str = " llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java/", #test suits
    benchmark_name: str = "humaneval",
    model_type: str = "deepseek-coder-6.7b-base",
    peft_type: str = "lora",
    train_dataset: str = "apr",
    ):
    validation_file = output_dir + '_'.join(model_type.split('-')) + '_' + peft_type + '_' + train_dataset + '_on_' + benchmark_name + '_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
 
    #count correct patches 
    plausible, total = 0, 0
    clean_and_create_benchmark_dir(benchmark_dir, benchmark_name)
    #load pacthes
    model_output = json.load(open(input_file, 'r'))

    validated_result = {}
    validated_result['patch_file'] = input_file
    validated_result['data'] = {}
    for proj in model_output['data']:
        
        print('start validating', proj)

        validated_result['data'][proj] = {}
        if 'output' not in model_output['data'][proj]:
            validated_result['data'][proj]['output'] = []
            continue
        total += 1

        current_is_correct = False
        #max{rank} == num_outputs == 10
        recopy_proj_code_and_test_case(benchmark_dir, proj, benchmark_name)
        for rank, patch in enumerate(model_output['data'][proj]['output']):
            if model_type == 'deepseek-coder-6.7b-base':
                end_bucket = patch.rfind('}')
                patch = patch[:end_bucket+1]
            buggy_code = open(benchmark_dir + f'src_bak/main/java/{benchmark_name}/buggy/' + proj + '.java', 'r').read()
            buggy_lines = buggy_code.split('\n')

            start_line_index = model_output['data'][proj]['function range'].split('-')[0]
            start_line_index = int(start_line_index.split(',')[0])
            end_line_index = model_output['data'][proj]['function range'].split('-')[1]
            end_line_index = int(end_line_index.split(',')[0])

            patch_prefix = '\n'.join(buggy_lines[:start_line_index-1])
            patch_suffix = '\n'.join(buggy_lines[end_line_index:])

            filename = benchmark_dir + f'src/main/java/{benchmark_name}/buggy/' + proj + '.java'

            open(filename, 'w').write(patch_prefix + '\n' + patch + '\n' + patch_suffix)
            correctness = humaneval_test_suite(proj, benchmark_dir)
            if correctness == 'plausible':
                if not current_is_correct:
                    plausible += 1
                    current_is_correct = True
                print(plausible, total, rank, "Plausible patch:", patch)
            elif correctness == 'wrong':
                print(plausible, total, rank, "Wrong patch:", patch)
            elif correctness == 'timeout':
                print(plausible, total, rank, "Timeout patch:", patch)
            elif correctness == 'uncompilable':
                print(plausible, total, rank, "Uncompilable patch:", patch)
            validated_result['data'][proj][rank] = correctness
            # shutil.copyfile(benchmark_dir + f'src_bak/main/java/{benchmark_name}/buggy/' + proj + '.java',
            #                 benchmark_dir + f'src/main/java/{benchmark_name}/buggy/' + proj + '.java')
        os.system(f'rm -rf {benchmark_dir}src/main/java/{benchmark_name}/buggy/*.java')
        os.system(f'rm -rf {benchmark_dir}src/test/java/{benchmark_name}/*.java')
        json.dump(validated_result, open(validation_file, 'w'), indent=2)
    print(f'results to file {validation_file}')
    # cal_result(validation_file, model_type, peft_type, train_dataset, benchmark_name)
    return validation_file
def humaneval_test_suite(algo, benchmark_dir):
    CUR_DIR = os.getcwd()
    try:
        os.chdir(benchmark_dir)
        out, err = exec_command_with_timeout(["mvn", "test", "-Dtest=TEST_" + algo.upper()], timeout=10)
        os.chdir(CUR_DIR)
        msg = (str(out) + str(err)).upper()
        if "compilation problems".upper() in msg or "compilation failure".upper() in msg:
            return 'uncompilable'
        elif "timeout".upper() in msg:
            return 'timeout'
        elif "build success".upper() in msg:
            return 'plausible'
        else:
            return "wrong"
    except Exception as e:
        print(e)
        os.chdir(CUR_DIR)
        return 'uncompilable'
if __name__ == '__main__':
    fire.Fire(validate_humaneval)
