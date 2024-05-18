import os
import gc
import torch
import tqdm as notebook_tqdm
import argparse
import pandas as pd

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import TrainerCallback, TrainerState, TrainerControl, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models

import wandb
#last_run_id = "kifgpikn"  # fetch the run_id from your wandb workspace
# resume the wandb run from the run_id
#run =wandb.init(entity="anushkayadav", project="huggingface", id=last_run_id, resume="must")

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control
    
base_model_id = "google/gemma-2b-it"

print("########## Model name : ", base_model_id)

#################################### Tokenizer ##############################################
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="right",
    add_eos_token=True,
    add_bos_token=True,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

####################################### Load Dataset #########################################
#--------DATASET--------
# Define the generate_prompt function
def generate_prompt(data_point):
    # Generate prompt
    instruction = "Please generate the tool planner answer for the question using tool calls in square brackets\n"
    prefix_text = 'Generate the Tool Planner Answer for the given question.\n\n'
    text = f"""<start_of_turn>user\n{prefix_text}{instruction}here is the question\n{data_point["Q"]} <end_of_turn>\n<start_of_turn>model\n{data_point["C"]} <end_of_turn>"""
    return text

train=pd.read_pickle("all_data_finals/tool_train.pkl")
eval=pd.read_pickle("all_data_finals/tool_val.pkl")

train_dataset =  Dataset.from_pandas(train)
eval_dataset = Dataset.from_pandas(eval)

# Apply the generate_prompt function and add the "prompt" column to the dataset
text_column = [generate_prompt(data_point) for data_point in train_dataset]
train_dataset = train_dataset.add_column("prompt", text_column)

# Apply the generate_prompt function and add the "prompt" column to the dataset
text_column = [generate_prompt(data_point) for data_point in eval_dataset]
eval_dataset = eval_dataset.add_column("prompt", text_column)

print("Train Dataset : ", train_dataset)
print("Eval Dataset : ", eval_dataset)

print('EXAMPLE')
print(train_dataset[0])

print("Dataset is loaded properly")


####################################### Load Model #########################################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                            quantization_config=bnb_config, 
                                            device_map={"": 0},
                                            use_cache=False)

model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)

print("Model Loaded")



# lora rank - 64, 7B - 24GB memory atleast, only Q, K, V, O, Grad Accumulation - 
config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        # "attn_out",
        # "ff_out",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.1,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print("Model Loaded with LoRA config")

# --gradient_checkpointing True, workers speed up processing,  grad accumulation - 8, 16, 22 (less memory)

####################################### Training Arguments #########################################
args=transformers.TrainingArguments(
    output_dir='./SAVED_GEMMA_2/',
    warmup_steps=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    group_by_length=True,
    bf16=False,
    num_train_epochs=5,
    learning_rate=5e-5,
    optim="paged_adamw_32bit",
    logging_strategy='steps',
    logging_steps=200,              # When to start reporting loss
    save_strategy='steps',       # Save the model checkpoint every logging step
    save_steps=400,                # Save checkpoints every 100 steps
    evaluation_strategy='steps', # Evaluate the model every epoch
    eval_steps=200,               # Evaluate and save checkpoints every 100 steps
    do_eval=True,                # Perform evaluation at the end of training
    report_to='wandb',           # Comment this out if you don't want to use weights & baises
    dataloader_pin_memory=True,                           
    dataloader_num_workers=4,
    dataloader_prefetch_factor=1,
    logging_first_step=True,
    lr_scheduler_type="cosine",
    seed=42,
    # bf16=True,
    # fp16=False,
    # tf32=True,
    disable_tqdm=False,
    run_name="tool_gemma2"
)

#5e-4, 5e-5
#another test
# def formatting_prompts_func(examples):
#     output_text = []
#     for i in range(len(examples["final_prompt"])):
#         prompt = examples["final_prompt"][i]
#         output_text.append(prompt)

#     return output_text



# ####################################### Data Collator #########################################
# # compute loss only for the summary, not for the prompt (conversation)
#response_template = " ### RESPONSE"
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

#response_template_with_context = "\n### RESPONSE"  # We added context here: "\n". This is enough for this tokenizer
#response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

#collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Seq Length - Based on distribution
####################################### SFT Training #########################################
print("################# Training started")
### SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=config,
    dataset_text_field="prompt",
    max_seq_length=2048, # input_args.max_seq_length,  # You can specify the maximum sequence length here
    tokenizer=tokenizer,
    args=args,
    packing=False,
    #formatting_func=formatting_prompts_func,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[SavePeftModelCallback()],
)
#trainer.train(resume_from_checkpoint='SAVED_MISTRAL_M1/checkpoint-4800/')
trainer.train()


print("################# Training is done")


trainer.model.save_pretrained(f"{args.output_dir}/final_checkpoint")
tokenizer.save_pretrained(f"{args.output_dir}/final_checkpoint")

# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()


#-------merge our adapter weights------------
# Reload model in FP16 (instead of NF4)
print("merge our adapter weights")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, model_id=f"{args.output_dir}/final_checkpoint")
model = model.merge_and_unload()

print("SAVING FINAL")
# Save model and tokenizer
model.save_pretrained(args.output_dir+"/sft")
tokenizer.save_pretrained(args.output_dir+"/sft")

print('SAVING DONE!')

# Flush memory
del model, base_model
gc.collect()
torch.cuda.empty_cache()