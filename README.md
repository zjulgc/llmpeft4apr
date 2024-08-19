# Exploring Parameter-Efficient Fine-Tuning of Large Language Model on Automated Program Repair

## Dependency

### Python

* Python 3.9.17
* PyTorch 2.0.1
* Huggingface transformers 4.35.2
* wandb
* pef 0.6.2

- accelerate 0.24.1

- datasets 2.13.0

- trl

- fire

* nvitop

### Others

- Java 8



## Content
The file structure of the artifact is as follow:

### APR-INSTRUCTION_construct;

- contains source code of constructing `APR-INSTRUCTION` ,base existing APR dataset[1]

### **codellama_7b_hf:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ and Full-model Fine-tuning

- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `codellama-7b-hf` and `codellama-7b-hf` with peft weights, validation results of generated pacthes

### **codellama_13b_hf:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ 
- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `codellama-13b-hf` and `codellama-13b-hf` with peft weights, validation results of generated pacthes

### **deepseek_coder_6.7b:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ 
- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `Deepseek-Coder Base 6.7B` and `Deepseek-Coder Base 6.7B` with peft weights, validation results of generated pacthes

### **llama2_7b_hf:**  

- **output:** peft weights by different peft method(lora, p-tuning，prefix tuning , $(IA)^3$ 
- **results:** results of generated pacthes on benchmarks(Humaneval-Java, Defect4j and Quixbugs) inferencing by `Llama-2-7b-hf` and `Llama-2-7b-hf` with peft weights, validation results of generated pacthes

### instruction_tuning_dataset

- Instruction Dataset used this paper
  - apr_instruction_30k.json: the APR instruction dataset constructed this paper
  - oss_instrcution_30k.json: 30k random selection of OSS-Instruction Dataset
  - code_alpaca_20k.json: Code Alpaca Instruction Dataset
  - The rest of data is used for RQ3 to explore the impact of training data size for performance, which is parted as 10k, 15k, 20k and 25k

### **inference_and_validation_src:**

- This directory consists of source code used for patches generation and validation of LLMs
    |  file name  |       description     |
    |  :----:             |       :----:          |
    | defects4j_patch_validate.py | patches generation and validation on Defects4j benchmark |
    | humaneval_patch_validate.py | patches generation and validation on Humaneval-Java benchmark |
    | quixbugs_patch_validate.py | patches generation and validation on Quixbugs benchmark |
    | peft_patch_validation.py | Entry of model validation with PEFT methods, and then select different scripts for verification |
    | fmft_generate_patch.py | Entry of model validation with Full-model fine-tuning, and then select different scripts for verification |
    | generate_patch_infill.py | Entry of CodeLlama 7b validation with no fine-tuning and infill templates, and then select different scripts for verification |
    | prompter.py | convert instances of benchmark to instruction |
    | result_look.py | record $pass@k$ of each validation |

### **inference_scripts:**

- This directory consists of bash scripts used for patches generation and validation of LLMs


- each script is formed as `model name`\_`Fine-tuning method`\_`instruction dataset of Fine-tuning`\_validation.sh

### **train_scripts:**

- This directory consists of bash scripts used for LLMs training 
- each script is formed as `model name`\_instrcution\_`Fine-tuning method`\_`hyper-parameters(Optional)`\_train\_`instruction dataset of Fine-tuning`\_validation.sh

### train_src:


- This directory consists of source code used for LLM trainnin

    |  file name    |       description     |
    |  :----:             |       :----:          |
    | sfttrain_peft.py | Training code for PEFT methods |
    | sfttrain_ft.py | Training code for Full-model Fine-tuning |
    |   prompter.py    |  Add additional prompt for instruction   |
    

### results_hyper_parameters:

- This directory consists of results of patches generation and validation in experiments of RQ3




## NOTICE  

- Due to the size of `Fine-tuning weights`  is too large, so we do not upload them on Github now
- Considering the anonymous review,  we will  release weights after review


## Cites  

```
[1] Zhu, Qihao, et al. "A syntax-guided edit decoder for neural program repair." Proceedings of the 29th ACM joint meeting on European software engineering conference and symposium on the foundations of software engineering. 2021.
```

