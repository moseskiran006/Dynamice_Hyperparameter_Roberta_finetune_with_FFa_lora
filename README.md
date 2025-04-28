# Federated Learning with LoRA for Natural Language Inference

## Project Overview
This project implements a federated learning system for **Natural Language Inference (NLI)** using the **Multi-Genre Natural Language Inference (MNLI)** dataset.  
It integrates several advanced techniques:

- **Federated Learning**: Trains a model across multiple clients without sharing raw data.
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning of large language models.
- **Non-IID Data Distribution**: Realistic client data partitioning using the Dirichlet distribution.
- **Dynamic Hyperparameter Adaptation**: Adjusts learning rates and LoRA ranks during training.

## Key Features
- Privacy-preserving distributed training
- Efficient parameter tuning with LoRA adapters
- Adaptive learning rate and rank adjustment
- Comprehensive evaluation on matched and mismatched validation sets
- GPU optimization with mixed precision training

## Requirements

### Python Libraries
Make sure the following packages are installed:

- `torch>=2.0.0`
- `transformers>=4.30.0`
- `datasets>=2.12.0`
- `peft>=0.4.0`
- `numpy>=1.23.0`
- `scikit-learn>=1.2.0`
- `matplotlib>=3.7.0`
- `tqdm>=4.65.0`

### Hardware Requirements
- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum**: 16GB RAM (for CPU execution)




##Clone the repository:
```bash
git clone https://github.com/yourusername/federated-lora-nli.git
cd federated-lora-nli
```



# For Linux/MacOS
```python -m venv venv
source venv/bin/activate
```
# For Windows
```bash
python -m venv venv
venv\Scripts\activate
```
##Install Libraries
```bash
pip install -r requirements.txt
```

##Run the model 
```bash
python federated_lora_nli.py
```

