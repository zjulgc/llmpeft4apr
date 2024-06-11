import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import inspect
import fire
import torch
import wandb
import sys
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import transformers
from datasets import load_dataset

from prompter import Prompter



def create_model_and_tokenizer(model_name_or_path, model_type, load_in_8bit=False):
    """Create model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
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


def train(
    # debug mode or not
    debug: bool = False,
    # model params
    model_type: str = "CodeLlama-7b-hf",
    model_name_or_path: str = '/home/survolt/warehouse/llmpeft4apr/models/incoder-6B',
    load_in_8bit: bool = False,
    save_dir: str = "./output",
    # data params
    dataset_dir: str = '/home/survolt/warehouse/llmpeft4apr/dataset/fine_tuning_data/instructions/',
    # use_oss_dataset: bool = False,
    oss_dataset_path: str = "oss_instruction.json",
    cache_dir: str="",
    max_seq_len: int = 1200, #  CUTOFF-LEN
    # val_set_size: int = 7000,
    # train hyper params
    # batch_size: int = 128,
    # micro_batch_size: int = 4,
    num_train_epochs: int = 3,
    train_batch_size: int = 3,
    valid_batch_size: int = 3,
    lr: float = 1e-5,
    peft_methods: str = "lora", # ['lora', 'p-tuning', 'prefix-tuning', 'IA3']
    # args for logging
    use_wandb: bool = False,
    wandb_project: str = "apr-peft",
    # fp16
    fp16: bool = False,
    bf16: bool = True,
    train_on_inputs: bool = False,
    gradient_accumulation_steps: int = 1,
):
    if debug:
        transformers.logging.set_verbosity_info()
    # add a time info to output dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=model_type+'_'+peft_methods,
            config=inspect.getargvalues(inspect.currentframe()).locals,
        )
        
        print(inspect.getargvalues(inspect.currentframe()).locals)
    #ensure model_name_or_path is not None
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # create base-model and tokenizer

    model, tokenizer = create_model_and_tokenizer(model_name_or_path, model_type, load_in_8bit)

    # create peft model
    # model.print_trainable_parameters()

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_seq_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    def generate_and_tokenize_prompt(data_point, add_eos_token=True):
        full_prompt = Prompter.generate_prompt(
            instruction=data_point["instruction"],
            label=data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = Prompter.generate_prompt(
                instruction=data_point["instruction"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    # load buggy dataset
    ds = load_dataset("json", data_files = dataset_dir + oss_dataset_path, cache_dir=cache_dir )

    ds = ds['train'].train_test_split(test_size=int(len(ds['train']) * 0.2), seed=42)
    ds['valid'] = ds['test']
    print("training dataset size:", len(ds['train']), "validation dataset size:", len(ds['valid']))
    if debug:
        ds['train'] = ds['train'].select(list(range(1024)))
        ds['valid'] = ds['valid'].select(list(range(32)))
    training_dataset = ds['train'].shuffle().map(generate_and_tokenize_prompt, remove_columns=["input", "instruction", "output"])
    validation_dataset = ds['valid'].shuffle().map(generate_and_tokenize_prompt, remove_columns=["input", "instruction", "output"])
    # input is empty, drop it.

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    # define trainer
    # total_step = (len(training_dataset) // (numof_gpus * per_device_train_batch_size) * num_train_epochs // gradient_accumulation_steps)
    # (128k // (1 * 64)) * 3 // 1 = 6k
    trainer = transformers.Trainer(
        model=model,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        args=transformers.TrainingArguments(
            num_train_epochs=num_train_epochs,
            output_dir=save_dir,
            overwrite_output_dir=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # args for build dataloader
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=valid_batch_size,
            dataloader_drop_last=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=20,
            # args for create_optimizer
            optim="adamw_hf", # other possible choices: ['adamw_torch', 'sgd', 'adagrad', 'adafactor'] adafactor will be deprecated in the future
            learning_rate=lr,  # use default values for other adam args
            # args for create_scheduler
            lr_scheduler_type="cosine",  # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
            warmup_ratio=0.05,
            # args for eval and save
            evaluation_strategy="steps",
            eval_steps=0.2,  # [0,1) means ratio, [1,~) means steps
            save_strategy="steps",
            save_steps=0.3,
            # load_best_model_at_end=True,
            logging_dir=save_dir,
            logging_steps=0.0005,
            group_by_length=True,
            report_to="wandb" if use_wandb else "none",
            run_name=model_type + '_' + peft_methods if use_wandb else None,
            fp16=fp16,
            bf16=bf16
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks=[PeftSavingCallback],
    )
    # accelerate training
    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        print("use torch.compile")
    # finally go train and eval
    trainer.train()
    model.save_pretrained(save_dir)


if __name__ == '__main__':
    fire.Fire(train)
