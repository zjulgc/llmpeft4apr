import fire
import datetime
from humaneval_patch_validate import validate_humaneval
from quixbugs_patch_validate import validate_quixbugs
from  defects4j_patch_validate import validate_defects4j
from result_look import cal_result

def peft_patch_validation(
    input_file: str = "/c21071/lgc/llmpeft4apr/results/deepseek_coder_6.7b_base_apr_p-tuning_on_humaneval_output_24_03_31_13_27_27.json", 
    output_dir: str = "/c21071/lgc/llmpeft4apr/results/", #validation results
    # copy from /c21071/lgc/llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java
    benchmark_dir: str = "/c21071/lgc/llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java/", #test suits
    benchmark_name: str = "humaneval",
    model_type: str = "deepseek-coder-6.7b-base",
    peft_type: str = "lora",
    train_dataset: str = "apr",
    validation_time: str = ""
):
    validate_fp = ''
    if benchmark_name == 'humaneval':
        validate_fp = validate_humaneval(
                input_file= input_file,
                output_dir=output_dir,
                benchmark_dir=benchmark_dir,
                benchmark_name=benchmark_name,
                model_type=model_type,
                peft_type=peft_type,
                train_dataset=train_dataset
            )
        
        # cal_result(validate_fp)
    elif benchmark_name == 'quixbugs':
        print(benchmark_dir)
        validate_fp =validate_quixbugs(
            input_file= input_file,
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            model_type=model_type,
            peft_type=peft_type,
            tmp_dir=benchmark_dir,
            train_dataset=train_dataset
        )
    elif benchmark_name == 'defects4j':
        print(validation_time[1:])
        validate_fp =validate_defects4j(
            input_file= input_file,
            output_dir=output_dir,
            benchmark_dir=benchmark_dir,
            benchmark_name=benchmark_name,
            model_type=model_type,
            peft_type=peft_type,
            tmp_dir=benchmark_dir,
            train_dataset=train_dataset,
            validation_time=validation_time[1:]
        )
    else:
        print('Wrong benchmark name!')
    if validate_fp != '':
        cal_result(validate_fp, model_type, peft_type, train_dataset, benchmark_name)
if __name__ == '__main__':
    fire.Fire(peft_patch_validation)