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
def find_last_closed_brace_index(s):
    stack = []
    end_flag = False
    last_closed_brace_index = -1

    for i, char in enumerate(s):
        if char == '{':
            end_flag = True
            stack.append(i)
        elif char == '}':
            if len(stack) == 0:
                return -1
            stack.pop()
            last_closed_brace_index = i
        if len(stack) == 0 and end_flag:
            return last_closed_brace_index
    return -1
def quixbugs_test_suite(algo, quixbugs_dir):
    QUIXBUGS_MAIN_DIR = quixbugs_dir
    CUR_DIR = os.getcwd()
    FNULL = open(os.devnull, 'w')
    JAR_DIR = 'libs/'
    try:
        os.chdir(QUIXBUGS_MAIN_DIR)
        p1 = subprocess.Popen(["javac", "-cp", ".:java_programs:" + JAR_DIR + "junit-4.12.jar:" + JAR_DIR +
                               "hamcrest-core-1.3.jar", "java_testcases/junit/" + algo.upper() + "_TEST.java"],
                              stdout=subprocess.PIPE, stderr=FNULL, universal_newlines=True)
        # print(f'javac -cp .:java_programs:{JAR_DIR}junit-4.12.jar:{JAR_DIR}hamcrest-core-1.3.jar java_testcases/junit/{algo.upper()}_TEST.java')
        out, err = exec_command_with_timeout(
            ["java", "-cp", ".:java_programs:" + JAR_DIR + "junit-4.12.jar:" + JAR_DIR + "hamcrest-core-1.3.jar",
             "org.junit.runner.JUnitCore", "java_testcases.junit." + algo.upper() + "_TEST"], timeout=5
        )
        # print(f'java -cp .:java_programs:{JAR_DIR}junit-4.12.jar:{JAR_DIR}hamcrest-core-1.3.jar org.junit.runner.JUnitCore java_testcases.junit.{algo.upper()}_TEST')

        os.chdir(CUR_DIR)
        if "FAILURES" in str(out) or "FAILURES" in str(err):
            return 'wrong'
        elif "TIMEOUT" in str(out) or "TIMEOUT" in str(err):
            return 'timeout'
        else:
            return 'plausible'
    except Exception as e:
        print(e)
        os.chdir(CUR_DIR)
        return 'uncompilable'
def compile_fix(filename, tmp_dir):
    FNULL = open(os.devnull, 'w')
    p = subprocess.call(["javac",
                         tmp_dir + "Node.java",
                         tmp_dir + "WeightedEdge.java",
                         filename],  stderr=FNULL)
    # print(f'javac {tmp_dir}Node.java {tmp_dir}WeightedEdge.java {filename}')
    # print(p)
    return False if p else True

def validate_quixbugs(
    #patches generated, json file
    #make sure format of  ouput patch
    input_file: str = "/c21071/lgc/llmpeft4apr/codellama_7b_hf/result/CodeLlama_7b_hf_lora_on_quixbugs_output_patches_1709040073.5193715.json", 
    output_dir: str = "/c21071/lgc/llmpeft4apr/codellama_7b_hf/result/", #validation results
    benchmark_name: str = "quixbugs",
    model_type: str = "CodeLlama-7b-hf",
    peft_type: str = "lora",
    tmp_dir: str = '/c21071/lgc/llmpeft4apr/validation_benchmark_dataset/benchmarks/quixbugs_tmp',
    train_dataset: str = "apr"
    ):
    validation_file = output_dir + '_'.join(model_type.split('-')) + '_' + peft_type + '_' + train_dataset + '_on_' + benchmark_name + '_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    # bug_loc_map = create_bug_loc_map(bug_locs_dir, benchmark_name)
    #count correct patches 
    plausible, total = 0, 0
    # if not os.path.exists(tmp_dir):
    #     exec_command_with_timeout(['mkdir', tmp_dir])

    #load pacthes
    model_output = json.load(open(input_file, 'r'))

    validated_result = {}
    validated_result['patch_file'] = input_file
    validated_result['data'] = {}
    for proj in model_output['data']:
        print('start validating', proj)

        exec_command_with_timeout((['rm', '-rf', tmp_dir + '/java_programs/']))
        exec_command_with_timeout((['mkdir', tmp_dir + '/java_programs/']))
        shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                        tmp_dir + "/java_programs/" + proj + '.java')
        shutil.copyfile(tmp_dir + "/java_programs_bak/Node.java", tmp_dir + "/java_programs/Node.java")
        shutil.copyfile(tmp_dir + "/java_programs_bak/WeightedEdge.java", tmp_dir + "/java_programs/WeightedEdge.java")

        validated_result['data'][proj] = {}
        total += 1

        current_is_correct = False
        error_compile = 0
        for rank, patch in enumerate(model_output['data'][proj]['output']):
            if model_output['data'][proj]['input'] == "":
                # print(proj)
                continue
            buggy_code_lines = model_output['data'][proj]['input'].strip().split('\n')
            bak_buggy_code = open(tmp_dir + "/java_programs_bak/" + proj + '.java', 'r').read()
            patch_prefix_start_index = bak_buggy_code.find(buggy_code_lines[0])
            patch_prefix_end_index = patch_prefix_start_index + find_last_closed_brace_index(bak_buggy_code[patch_prefix_start_index:])
             

            filename = tmp_dir + "/java_programs/" + proj + '.java'
            open(filename, 'w').write(bak_buggy_code[:patch_prefix_start_index] + patch + bak_buggy_code[patch_prefix_end_index+1:])
            compile = compile_fix(filename, tmp_dir + "/java_programs/")

            correctness = 'uncompilable'

            if compile:
                correctness = quixbugs_test_suite(proj, quixbugs_dir=tmp_dir)
                if correctness == 'plausible':
                    if not current_is_correct:
                        plausible += 1
                        current_is_correct = True
                    print(plausible, total, rank, "Plausible patch:", patch)
                elif correctness == 'wrong':
                    print(plausible, total, rank, "Wrong patch:", patch)
                elif correctness == 'timeout':
                    print(plausible, total, rank, "Timeout patch:", patch)
            else:
                print(plausible, total, rank, 'Uncompilable patch:', patch)
            validated_result['data'][proj][rank] = correctness
            shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                            tmp_dir + "/java_programs/" + proj + '.java')
        json.dump(validated_result, open(validation_file, 'w'), indent=2)
    print(f'results to file {validation_file}')
    return validation_file
if __name__ == '__main__':
    fire.Fire(validate_quixbugs)
