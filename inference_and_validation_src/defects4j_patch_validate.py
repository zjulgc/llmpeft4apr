import json
import sys
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

def defects4j_test_suite(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = exec_command_with_timeout(["defects4j", "test", "-r"], timeout)
    return out, err

def checkout_defects4j_project(project, bug_id, tmp_dir):
    FNULL = open(os.devnull, 'w')
    command = "defects4j checkout " + " -p " + project + " -v " + bug_id + " -w " + tmp_dir
    p = subprocess.Popen([command], shell=True, stdout=FNULL, stderr=FNULL)
    p.wait()
def clean_tmp_folder(tmp_dir):
    if os.path.isdir(tmp_dir):
        for files in os.listdir(tmp_dir):
            file_p = os.path.join(tmp_dir, files)
            try:
                if os.path.isfile(file_p):
                    os.unlink(file_p)
                elif os.path.isdir(file_p):
                    shutil.rmtree(file_p)
            except Exception as e:
                print(e)
    else:
        os.makedirs(tmp_dir)
def compile_fix(project_dir):
    os.chdir(project_dir)
    p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if "FAIL" in str(err) or "FAIL" in str(out):
        return False
    return True
def defects4j_trigger(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = exec_command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout)
    return out, err
def defects4j_test_one(project_dir, test_case, timeout=300):
    os.chdir(project_dir)
    out, err = exec_command_with_timeout(["defects4j", "test", "-t", test_case], timeout)
    return out, err
def validate_defects4j(
    #patches generated, json file
    #make sure format of  ouput patch
    input_file: str = "/home/survolt/warehouse/llmpeft4apr/codellama_result/CodeLlama_7b_hf_output_humaneval_infill.json", 
    output_dir: str = "/home/survolt/warehouse/llmpeft4apr/codellama_result/", #validation results
    benchmark_dir: str = "/home/survolt/warehouse/llmpeft4apr/dataset/benchmarks/humaneval-java/", #test suits
    benchmark_name: str = "defects4j",
    model_type: str = "CodeLlama-7b-hf",
    peft_type: str = "lora",
    tmp_dir: str = ' llmpeft4apr/validation_benchmark_dataset/benchmarks/tmp_dir/',
    train_dataset: str = "apr",
    validation_time: str = ""
    ):
    validation_file = output_dir + '_'.join(model_type.split('-')) + '_' + peft_type + '_' + train_dataset + '_on_' + benchmark_name + '_validation_' + str(validation_time) + '.json'
    print(validation_file)
    # return None
    #count correct patches 
    plausible, total = 0, 0
    #check tmp_dir 
    if not os.path.exists(tmp_dir):
        exec_command_with_timeout(['mkdir', tmp_dir])

    # clean_and_create_benchmark_dir(benchmark_dir, benchmark_name)
    #load pacthes
    model_output = json.load(open(input_file, 'r'))

    validated_result = {}
    if os.path.exists(validation_file):
        print('validation file existed!')
        validated_result = json.load(open(validation_file, 'r'))
        total = len(validated_result['data'].keys())
        for proj, res_dict in validated_result['data'].items():
            for r, val_res in res_dict.items():
                if val_res == 'plausible':
                    plausible += 1
                    break
    else:
        print('No existing validation file!')
        validated_result['patch_file'] = input_file
        validated_result['data'] = {}
    
    for k in model_output['data']:
        print('start validating', k)
        if k in validated_result['data']:
            print('validation result existed!')
            continue
        if 'output' not in model_output['data'][k]:
            continue
        key_list = k.split('_')

        proj, bug_id, bug_line_loc = key_list[0], key_list[1], key_list[-1]
        path = '_'.join(key_list[2: -1])
        if path[0] == '/':
            path = path[1:]
        function_start_loc, function_end_loc = model_output['data'][k]['function range'].split('-')
        function_start_loc = int(function_start_loc.split(',')[0])
        function_end_loc = int(function_end_loc.split(',')[0])

        
        
        total += 1
        validated_result['data'][k] = {}
        
        clean_tmp_folder(tmp_dir)

        checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)

        buggy_code_file = tmp_dir + path
        buggy_code_str = open(buggy_code_file, 'r').read()
        buggy_code_lines = buggy_code_str.split('\n')

        if proj == "Mockito":
            print("Mockito needs separate compilation")
            compile_fix(tmp_dir)

        start_time = time.time()
        init_out, init_err = defects4j_test_suite(tmp_dir)
        standard_time = int(time.time() - start_time)

        failed_test_cases = str(init_out).split(' - ')[1:]
        for i, failed_test_case in enumerate(failed_test_cases):
            failed_test_cases[i] = failed_test_case.strip()
        init_fail_num = len(failed_test_cases)
        print(init_fail_num, str(standard_time) + 's')

        trigger, err = defects4j_trigger(tmp_dir)
        triggers = trigger.strip().split('\n')
        for i, trigger in enumerate(triggers):
            triggers[i] = trigger.strip()
        print('trigger number:', len(triggers))
        current_is_correct = False
        # recopy_proj_code_and_test_case(benchmark_dir, proj, benchmark_name)
        for rank, patch in enumerate(model_output['data'][k]['output']):
            #find buggy code file and write fix pacthes
            shutil.copyfile(buggy_code_file, buggy_code_file + '.bak')

            fix_code_liens = buggy_code_lines[:function_start_loc-1] + patch.split('\n') + buggy_code_lines[function_end_loc:]

            open(buggy_code_file, 'w').write('\n'.join(fix_code_liens))

            if proj == "Mockito":
                compile_fix(tmp_dir)
            outs = []
            correctness = None
            start_time = time.time()
            if standard_time >= 10 and len(triggers) <= 5:
                for trigger in triggers:
                    out, err = defects4j_test_one(tmp_dir, trigger, timeout=min(300, int(2*standard_time)))
                    if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                        print(plausible, total, rank, 'Time out for patch: ', patch,
                            str(int(time.time() - start_time)) + 's')
                        correctness = 'timeout'
                        break
                    elif 'FAIL' in str(err) or 'FAIL' in str(out):
                        print(plausible, total, rank, 'Uncompilable patch:', patch,
                            str(int(time.time() - start_time)) + 's')
                        correctness = 'uncompilable'
                        break
                    elif "Failing tests: 0" in str(out):
                        continue
                    else:
                        outs += str(out).split(' - ')[1:]
            if len(set(outs)) >= len(triggers):
                # does not pass any one more
                print(plausible, total, rank, 'Wrong patch:', patch,
                    str(int(time.time() - start_time)) + 's')
                correctness = 'wrong'

            if correctness is None:
                # pass at least one more trigger case
                # have to pass all non-trigger
                out, err = defects4j_test_suite(tmp_dir, timeout=min(300, int(2*standard_time)))

                if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                    print(plausible, total, rank, 'Time out for patch: ', patch,
                        str(int(time.time() - start_time)) + 's')
                    correctness = 'timeout'
                elif 'FAIL' in str(err) or 'FAIL' in str(out):
                    print(plausible, total, rank, 'Uncompilable patch:', patch,
                        str(int(time.time() - start_time)) + 's')
                    correctness = 'uncompilable'
                elif "Failing tests: 0" in str(out):
                    if not current_is_correct:
                        current_is_correct = True
                        plausible += 1
                    print(plausible, total, rank, 'Plausible patch:', patch,
                        str(int(time.time() - start_time)) + 's')
                    correctness = 'plausible'
                elif len(str(out).split(' - ')[1:]) < init_fail_num:
                    # fail less, could be correct
                    current_failed_test_cases = str(out).split(' - ')[1:]
                    no_new_fail = True
                    for current_failed_test_case in current_failed_test_cases:
                        if current_failed_test_case.strip() not in failed_test_cases:
                            no_new_fail = False
                            break
                    if no_new_fail:
                        # fail less and no new fail cases, could be plausible
                        if not current_is_correct:
                            current_is_correct = True
                            plausible += 1
                        print(plausible, total, rank, 'Plausible patch:', patch,
                                str(int(time.time() - start_time)) + 's')
                        correctness = 'plausible'
                    else:
                        print(plausible, total, rank, 'Wrong patch:', patch,
                                str(int(time.time() - start_time)) + 's')
                        correctness = 'wrong'
                else:
                    print(plausible, total, rank, 'Wrong patch:', patch,
                        str(int(time.time() - start_time)) + 's')
                    correctness = 'wrong'
            validated_result['data'][k][str(rank)] = correctness
            shutil.copyfile(buggy_code_file + '.bak', buggy_code_file)

        json.dump(validated_result, open(validation_file, 'w'), indent=2)

    json.dump(validated_result, open(validation_file, 'w'), indent=2)
    print(f'results to file {validation_file}')
    return validation_file


if __name__ == '__main__':
    fire.Fire(validate_defects4j)