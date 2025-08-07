import torch, os, multiprocessing
from datasets import load_dataset
from peft import OFTConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import argparse
from accelerate import PartialState

from transformers import set_seed

#use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

if torch.cuda.is_available():
    torch.set_autocast_enabled(True)


def main(model_path):
    model_name =  model_path.split('/')[-1]
    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    instruction_template = "### Human:"
    response_template = "### Assistant:"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    ds = load_dataset("timdettmers/openassistant-guanaco") #, cache_dir="/tmp/hf_cache")
    #Add the EOS token
    def process(row):
        row["text"] = row["text"]+"<|end_of_text|>"
        return row

    ds = ds.map(
        process,
        num_proc= 8,
        load_from_cache_file=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map={"": PartialState().process_index},
        torch_dtype=compute_dtype,
        cache_dir="/tmp/hf_cache",
    )
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=False)
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        
    output_dir = f"{model_name}"
    peft_config = OFTConfig(
        inference_mode=False,
        oft_block_size=32,
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    )

    model = get_peft_model(model, peft_config) #, autocast_adapter_dtype=False)

    training_arguments = SFTConfig(
        output_dir="./"+output_dir,
        # optim="adamw_8bit",
        optim="adamw_torch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        # log_level="debug",
        save_strategy="no",
        eval_strategy="no",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        max_steps=1000,
        # num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=2048,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        peft_config=peft_config,
        # tokenizer=tokenizer,
        args=training_arguments,
        data_collator=collator,
    )

    trainer.model.print_trainable_parameters()

    print("Starting standard training run...")
    trainer.train(resume_from_checkpoint=None)

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # accelerator = Accelerator()
    set_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    
    main(args.model_path)
