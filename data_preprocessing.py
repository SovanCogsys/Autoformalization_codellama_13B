import json
import glob
import random
import os

# === Configuration ===
input_dir = "/path/to/original/jsonl/files"
intermediate_txt_path = "/path/to/output/train_dataset.txt"
final_jsonl_path = "/path/to/output/train_dataset.jsonl"

def convert_to_instruction_format(input_files):
    converted = []

    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                user_prompt = entry["input"].strip()
                model_answer = entry["output"].strip()

                if "to Lean" in user_prompt:
                    system_prompt = "You are a helpful assistant that translates natural language statements into Lean 4 theorems."
                elif "to Isabelle" in user_prompt:
                    system_prompt = "You are a helpful assistant that translates natural language statements into Isabelle theorems."
                else:
                    raise ValueError(f"Unknown target language in: {user_prompt}")

                formatted = (
                    "<s>[INST] <<SYS>>\n"
                    f"{system_prompt}\n"
                    "<</SYS>>\n\n"
                    f"{user_prompt} [/INST] {model_answer} </s>"
                )
                converted.append(formatted)

    random.shuffle(converted)
    return converted

def write_txt_file(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def convert_txt_to_jsonl(txt_path, jsonl_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    examples = []
    blocks = raw_data.split("<s>")
    for block in blocks:
        if "</s>" in block:
            content = block.split("</s>")[0]
            full_block = f"<s>{content}</s>"
            cleaned = "\n".join(line.strip() for line in full_block.splitlines() if line.strip())
            json_line = json.dumps({"text": cleaned}, ensure_ascii=False)
            examples.append(json_line)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for line in examples:
            f.write(line + "\n")

if __name__ == "__main__":
    # Collect all JSONL files from the directory
    input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    # Step 1: Convert to instruction-tuned format
    formatted_data = convert_to_instruction_format(input_files)

    # Step 2: Write intermediate .txt file
    write_txt_file(formatted_data, intermediate_txt_path)

    # Step 3: Convert .txt file to JSONL
    convert_txt_to_jsonl(intermediate_txt_path, final_jsonl_path)

    print(f"✅ Conversion complete.\n- .txt saved to: {intermediate_txt_path}\n- .jsonl saved to: {final_jsonl_path}")
