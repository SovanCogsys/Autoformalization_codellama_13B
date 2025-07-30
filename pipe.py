import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig,PeftModel   
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

print(f"Number of CUDA devices: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")



# %%
#Load dataset
data_files = {
    "train": "/home/sovansahoo23/Ft/Final_set/train_f2 copy.jsonl",
    "validation": "/home/sovansahoo23/Ft/Final_set/valid_f copy.jsonl"
   
}

dataset = load_dataset("json", data_files=data_files)


model_name = "codellama/CodeLlama-13b-Instruct-hf"


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



#Step 2: Define a preprocessing function to tokenize and prepare labels
def preprocess_function(example):
    full_text = example["text"]

    # Tokenize full input text with padding/truncation
    encoding = tokenizer(
        full_text,
        max_length=650,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    #Prepare labels (copy input_ids)
    labels = input_ids.copy()

    #Mask out prompt tokens in labels with -100
    try:
        prompt = full_text.split("[/INST]")[0] + "[/INST]"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = [-100] * prompt_len
    except Exception as e:
        print(f"Prompt masking failed for input: {full_text}\nError: {e}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

#Step 3: Map the preprocessing function over the dataset
train_dataset = dataset.map(
    preprocess_function,
    remove_columns=["text"],  # remove original 'text' column if not needed
    batched=False,
)
#Step 4: Set format to PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# %%
model_name = "codellama/CodeLlama-13b-Instruct-hf"
# Set compute dtype for quantization
bnb_4bit_compute_dtype = "bfloat16"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

#Configure BitsAndBytes (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # If using quantization, you could enable this:
    #quantization_config=bnb_config,
    device_map={"": 0},  # Automatically place model on GPU 0
    torch_dtype=torch.bfloat16  # Use bfloat16 for reduced memory usage
    #cache_dir=TRANSFORMERS_CACHE
)

# Configuration adjustments
model.config.use_cache = False
model.config.pretraining_tp = 1




# %%
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=32,
    #32,
    lora_dropout=0.1,
    r=128,
    #128,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # explicit targeting

)
# 1. Define EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.0005
)

# Define training configuration for SFTTrainer
config = SFTConfig(
    output_dir="/scratch/work/sovan/output",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    #optim="paged_adamw_32bit",
    optim="adamw_torch",
    save_steps=5000,#
    logging_steps=10,
    learning_rate=9e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,#
    max_steps=-1,#
    #warmup_steps=
    warmup_ratio=0.03,#
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to=["tensorboard"], #        # accepts list
    dataset_text_field="text",         # field name containing text in dataset
    max_seq_length=650,#               # optional but helpful
    completion_only_loss=False,#
    bf16_full_eval=True,#
    eval_strategy="steps",
    eval_steps=5000,
    per_device_eval_batch_size=8,#
    dataloader_pin_memory=False,#
    dataloader_num_workers=0,#
    dataset_kwargs={"skip_prepare_dataset": True},
    label_names=["labels"],#
    save_total_limit=5,
    load_best_model_at_end=False,#asperMMA paper
    metric_for_best_model="eval_loss",
    greater_is_better = False,
    #logging_dir="/scratch/work/sovan/output",
  

)


# Initialize trainer with model, datasets, PEFT config, and training args
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset['train'],
    eval_dataset=train_dataset['validation'],
    peft_config=peft_config,
    args=config ,
    callbacks=[early_stopping]  
)

# Move trainer model to bfloat16 precision (if required)
trainer.model = trainer.model.to(torch.bfloat16)

# %%
trainer.train()

# %%
tokenizer.save_pretrained("/scratch/work/sovan/output/finetuned_model_8")


# %%
trainer.model.save_pretrained("/scratch/work/sovan/output/finetuned_model_8")



