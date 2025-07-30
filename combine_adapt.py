# %%
import os
import torch
from datasets import load_dataset
# from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer
)
from peft import LoraConfig,PeftModel   
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set the new cache directory

# Clear any cached CUDA memory
torch.cuda.empty_cache()



# %%
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

# %%
# The model that you want to train from the Hugging Face hub
model_name = "/scratch/work/sovan/huggingface/CodeLlama-13b-Instruct-hf"

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
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # If using quantization, you could enable this:
    # quantization_config=bnb_config,
    device_map={"": 0},  # Automatically place model on GPU 0
    torch_dtype=torch.bfloat16
)
adapter_path = "/scratch/work/sovan/output/finetuned_model_7"

model = PeftModel.from_pretrained(base_model, adapter_path)

model.config.use_cache = True         # Improves generation speed
model.config.pretraining_tp = 1       # Still 1 if no tensor parallelism is used


# %%

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # same as training
tokenizer.padding_side = "right"           # same as training

# Construct an inference-time prompt (same structure as training)
# prompt = """<s>[INST] <<SYS>>
# You are a helpful assistant that translates natural language statements into Isabelle theorems.
# <</SYS>>
# Statement in natural language:
# If we have a multiset 's' of functions from 'α' to 'M', and every function in 's' is strongly measurable, then the function that maps 'x' to the product of the multiset 's' mapped with the function 'f' from 'α' to 'M' at 'x', is also strongly measurable.
# Translate the statement in natural language to Lean: [/INST] """
prompt = """<s>[INST] <<SYS>>
You are a helpful assistant that translates natural language statements into Lean 4 theorems.
<</SYS>>
Statement in natural language:
All artists are sensitive.Some sensitive people are intelligent.No intelligent person is arrogant.
Then Some sensitive people are not arrogant.
Translate the above statement in natural language to Lean4 theorem: [/INST] """

# Tokenize (no labels needed)
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=650).to(model.device)

# %%
# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=520,
        do_sample=False,          # or True if you want sampling
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )



# %%
# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# %%
# Extract only generated response after prompt
response = result.split("[/INST]")[-1].strip()
print(response)



