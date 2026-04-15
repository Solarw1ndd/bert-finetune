from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


lora_config = LoraConfig(
    r=8,                    
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  


dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")

def format_prompt(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

dataset = dataset.map(format_prompt)


args = TrainingArguments(
    output_dir="./lora-results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    formatting_func=lambda x:x["text"],
)

trainer.train()
print("LoRA Training Completed!")


from peft import PeftModel
model.eval()
inputs = tokenizer("### Instruction:\nWhat is the capital of France?\n\n### Response:\n", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))