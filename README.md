# LLM Fine-tuning Projects

Two fine-tuning experiments using HuggingFace, PyTorch, and PEFT.

---

## Project 1: BERT Sentiment Classification
Fine-tuned BERT on the IMDB dataset for binary sentiment classification.

### Results
- Test Accuracy: 89%
- Model: bert-base-uncased
- Training data: 2000 samples, 2 epochs, fp16

### How to Run
pip install transformers datasets scikit-learn accelerate
python train.py

---

## Project 2: LoRA Fine-tuning on Qwen2.5-0.5B
Applied LoRA (Low-Rank Adaptation) to fine-tune Qwen2.5-0.5B-Instruct
on the Alpaca instruction-following dataset.

### Key Stats
- Trainable parameters: 540,672 / 494,573,440 (only 0.1%!)
- Method: LoRA with r=8, target modules: q_proj, v_proj
- Dataset: yahma/alpaca-cleaned (500 samples)

### How to Run
pip install transformers peft trl datasets accelerate
python lora_train.py

---

## Tech Stack
- PyTorch
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)
- GPU: NVIDIA RTX 5080 Laptop