

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import matplotlib.pyplot as plt
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
# Import AdamW from torch.optim instead of transformers
from torch.optim import AdamW
from datasets import load_dataset
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
import copy
import json
import warnings
import gc
warnings.filterwarnings("ignore")

# Check if GPU is available and set it up properly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Ensure GPU is being used by running a quick test
    test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(f"Test tensor device: {test_tensor.device}")
    
    # Set higher precision for numerical stability
    torch.backends.cuda.matmul.allow_tf32 = True  # Only for Ampere+ GPUs
else:
    device = torch.device("cpu")
    print("No GPU detected, using CPU instead.")

# Function to clear GPU cache
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory cleared. Currently allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Configuration
class FedConfig:
    num_clients = 3
    communication_rounds = 15
    local_epochs = 2
    batch_size = 16
    val_batch_size = 32
    max_seq_length = 128
    model_name = "roberta-base"
    
    # Initial hyperparameters (will be adjusted dynamically)
    init_learning_rate = 5e-5
    init_lora_rank = 4
    init_dp_noise_scale = 0.1  # For DP if implemented
    
    # Dynamic adaptation thresholds
    lr_decay_factor = 0.5
    lr_patience = 2
    rank_increase_step = 2
    
    # Lora configuration
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = ["query", "key", "value"]  # RoBERTa attention components for LoRA
    
    # Dataset related
    dataset_name = "multi_nli"
    num_labels = 3
    
    # Non-IID distribution parameters (adjust these to control skew)
    # Higher alpha = more balanced
    dirichlet_alpha = 0.5
    
    # GPU optimization
    use_mixed_precision = True  # Use mixed precision training if on GPU

# Enable mixed precision training if GPU is available
if torch.cuda.is_available() and FedConfig.use_mixed_precision:
    print("Enabling mixed precision training for faster GPU processing")
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

print("\n*** Loading tokenizer and preparing for data processing ***")
# Load & Preprocess Data
tokenizer = RobertaTokenizer.from_pretrained(FedConfig.model_name)

def preprocess_function(examples):
    """Tokenize and prepare examples for MNLI dataset"""
    # Get premise and hypothesis
    premises = examples["premise"]
    hypotheses = examples["hypothesis"]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        premises, 
        hypotheses, 
        max_length=FedConfig.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Add labels
    tokenized_inputs["labels"] = examples["label"]
    
    return tokenized_inputs

class MNLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples["input_ids"])
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.examples["input_ids"][idx],
            "attention_mask": self.examples["attention_mask"][idx],
            "labels": self.examples["labels"][idx]
        }

print("\n*** Loading MNLI dataset ***")
mnli_dataset = load_dataset(FedConfig.dataset_name)
print("Dataset loaded. Processing...")

train_data = mnli_dataset["train"]
validation_matched = mnli_dataset["validation_matched"]
validation_mismatched = mnli_dataset["validation_mismatched"]

# Preprocess the datasets
print("Tokenizing and preprocessing the data...")
train_processed = preprocess_function(train_data)
validation_matched_processed = preprocess_function(validation_matched)
validation_mismatched_processed = preprocess_function(validation_mismatched)

# Create PyTorch Datasets
train_dataset = MNLIDataset(train_processed)
val_matched_dataset = MNLIDataset(validation_matched_processed)
val_mismatched_dataset = MNLIDataset(validation_mismatched_processed)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation matched examples: {len(val_matched_dataset)}")
print(f"Validation mismatched examples: {len(val_mismatched_dataset)}")

# Create validation dataloader for central evaluation
val_dataloader = DataLoader(
    val_matched_dataset,
    batch_size=FedConfig.val_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available()  # Pin memory for faster data transfer to GPU
)

# Distribute data across clients (non-IID)
def distribute_data_dirichlet(dataset, num_clients, alpha):
    """
    Split data among clients according to a Dirichlet distribution
    Lower alpha = more skewed distribution (non-IID)
    """
    # Get labels for entire dataset
    all_labels = np.array(dataset.examples["labels"])
    num_classes = FedConfig.num_labels
    
    # Get indices for each class
    class_indices = [np.where(all_labels == class_id)[0] for class_id in range(num_classes)]
    
    # Initialize client data indices
    client_indices = [[] for _ in range(num_clients)]
    
    # Distribute class indices to clients according to Dirichlet distribution
    for class_id in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()  # Normalize
        
        # Calculate number of samples per client for this class
        samples_per_client = (proportions * len(class_indices[class_id])).astype(int)
        samples_per_client[-1] = len(class_indices[class_id]) - np.sum(samples_per_client[:-1])  # Ensure all samples are used
        
        # Shuffle indices for this class
        indices_for_class = class_indices[class_id].copy()
        np.random.shuffle(indices_for_class)
        
        # Distribute to clients
        start_idx = 0
        for client_id in range(num_clients):
            client_indices[client_id].extend(
                indices_for_class[start_idx:start_idx + samples_per_client[client_id]]
            )
            start_idx += samples_per_client[client_id]
    
    # Create subset datasets for each client
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    # Calculate distribution statistics for logging
    client_stats = []
    for client_id, indices in enumerate(client_indices):
        client_labels = [all_labels[idx] for idx in indices]
        dist = np.bincount(client_labels, minlength=num_classes) / len(client_labels)
        client_stats.append({
            "client_id": client_id,
            "num_samples": len(indices),
            "class_distribution": dist.tolist()
        })
    
    return client_datasets, client_stats

print("\n*** Distributing data to clients ***")
# Distribute data to clients using Dirichlet distribution for non-IID
client_datasets, client_stats = distribute_data_dirichlet(
    train_dataset, 
    FedConfig.num_clients, 
    FedConfig.dirichlet_alpha
)

# Print data distribution stats
print("\nData distribution across clients:")
for client_stat in client_stats:
    print(f"Client {client_stat['client_id']}: {client_stat['num_samples']} samples")
    print(f"Class distribution: {[f'{p:.3f}' for p in client_stat['class_distribution']]}")

# Clear memory before model creation
clear_gpu_memory()

print("\n*** Creating model with LoRA adapters ***")
# Define global model with LoRA configuration
def create_model_with_lora(lora_rank=FedConfig.init_lora_rank):
    # Load base model
    model = RobertaForSequenceClassification.from_pretrained(
        FedConfig.model_name,
        num_labels=FedConfig.num_labels
    )
    
    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,  # Low-rank dimension
        lora_alpha=FedConfig.lora_alpha,
        lora_dropout=FedConfig.lora_dropout,
        target_modules=FedConfig.target_modules,
        bias="none"
    )
    
    # Apply LoRA adapter to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

# Create the global model
global_model = create_model_with_lora()
global_model.to(device)
print(f"Global model created with LoRA adapter and moved to {device}")
print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Client training function
def train_client(client_id, model, dataset, config, current_hyperparams):
    """Client-side training with dynamic hyperparameter adaptation"""
    
    # Set client model to training mode
    model.train()
    
    # Extract current hyperparameters
    learning_rate = current_hyperparams["learning_rate"]
    lora_rank = current_hyperparams["lora_rank"]
    
    # For tracking training progress
    running_losses = []
    patience_counter = 0
    best_loss = float('inf')
    
    # Create data loader for client dataset with GPU optimizations
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * config.local_epochs
    )
    
    # Training loop
    for epoch in range(config.local_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Client {client_id} - Epoch {epoch+1}/{config.local_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                # Standard training path (CPU or GPU without mixed precision)
                outputs = model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Update stats
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        running_losses.append(avg_epoch_loss)
        print(f"Client {client_id} - Epoch {epoch+1}/{config.local_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        
        # Dynamic hyperparameter adaptation
        if epoch > 0:
            # If loss is not improving, reduce learning rate
            if avg_epoch_loss >= best_loss:
                patience_counter += 1
                if patience_counter >= config.lr_patience:
                    new_lr = learning_rate * config.lr_decay_factor
                    print(f"Client {client_id} - Reducing learning rate from {learning_rate:.2e} to {new_lr:.2e}")
                    learning_rate = new_lr
                    current_hyperparams["learning_rate"] = learning_rate
                    
                    # Update optimizer with new learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    
                    patience_counter = 0
            else:
                patience_counter = 0
                best_loss = avg_epoch_loss
    
    # Analyze convergence speed
    if len(running_losses) > 1:
        # If loss reduction is small, consider increasing LoRA rank
        loss_improvement = (running_losses[0] - running_losses[-1]) / running_losses[0]
        if loss_improvement < 0.1:  # Less than 10% improvement
            new_rank = lora_rank + config.rank_increase_step
            print(f"Client {client_id} - Convergence is slow. Increasing LoRA rank from {lora_rank} to {new_rank}")
            current_hyperparams["lora_rank"] = new_rank
    
    # Return updated model and hyperparameters
    return model, current_hyperparams

# Server aggregation function
def aggregate_models(global_model, client_models, client_hyperparams):
    """Aggregate client models using Federated Averaging (FedAvg)"""
    global_dict = global_model.state_dict()
    
    # Count total clients
    num_clients = len(client_models)
    
    # Simple averaging of model parameters
    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            # Initialize parameter sum
            global_dict[k] = torch.zeros_like(global_dict[k])
            
            # Sum parameters from all clients
            for client_model in client_models:
                global_dict[k] += client_model.state_dict()[k] / num_clients
    
    # Load updated parameters into global model
    global_model.load_state_dict(global_dict)
    
    # Aggregate hyperparameters (take mean)
    agg_hyperparams = {
        "learning_rate": sum(params["learning_rate"] for params in client_hyperparams) / num_clients,
        "lora_rank": round(sum(params["lora_rank"] for params in client_hyperparams) / num_clients)
    }
    
    return global_model, agg_hyperparams

# Evaluation function
def evaluate_model(model, dataloader):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Use mixed precision for inference if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss
                
            total_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1
    }

# Implementation of federated learning process
def run_federated_learning():
    """Main federated learning loop"""
    # Initialize global model
    print("\n*** Starting Federated Learning Process ***")
    global_model = create_model_with_lora()
    global_model.to(device)
    
    # Initialize metrics tracking
    results = {
        "rounds": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "hyperparams": []
    }
    
    # Initialize client hyperparams tracking
    client_hyperparams = []
    for client_id in range(FedConfig.num_clients):
        client_hyperparams.append({
            "learning_rate": FedConfig.init_learning_rate,
            "lora_rank": FedConfig.init_lora_rank,
            "dp_noise_scale": FedConfig.init_dp_noise_scale
        })
    
    # Initial evaluation
    print("\nInitial evaluation:")
    metrics = evaluate_model(global_model, val_dataloader)
    print(f"Initial model - Val Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    results["rounds"].append(0)
    results["val_loss"].append(metrics["loss"])
    results["val_accuracy"].append(metrics["accuracy"])
    results["val_f1"].append(metrics["f1_score"])
    results["hyperparams"].append(client_hyperparams.copy())
    
    # Federated learning rounds
    for round_num in range(1, FedConfig.communication_rounds + 1):
        print(f"\n--- Communication Round {round_num}/{FedConfig.communication_rounds} ---")
        
        # Track client models and new hyperparameters for this round
        client_models = []
        updated_hyperparams = []
        
        # Train each client with its local data
        for client_id in range(FedConfig.num_clients):
            print(f"\nTraining Client {client_id}")
            print(f"Hyperparameters: LR={client_hyperparams[client_id]['learning_rate']:.2e}, Rank={client_hyperparams[client_id]['lora_rank']}")
            
            # Clear GPU memory before each client training
            clear_gpu_memory()
            
            # Create copy of global model for client
            client_model = copy.deepcopy(global_model)
            
            # Potentially update LoRA rank if it has changed
            if round_num > 1 and client_hyperparams[client_id]["lora_rank"] != FedConfig.init_lora_rank:
                old_rank = FedConfig.init_lora_rank
                new_rank = client_hyperparams[client_id]["lora_rank"]
                
                # Need to create a fresh model with the new rank
                if old_rank != new_rank:
                    print(f"Client {client_id} - Updating LoRA rank from {old_rank} to {new_rank}")
                    client_model = create_model_with_lora(new_rank)
                    client_model.to(device)
                    
                    # Copy non-LoRA parameters from global model
                    global_dict = global_model.state_dict()
                    client_dict = client_model.state_dict()
                    
                    for k in client_dict.keys():
                        if "lora" not in k:
                            client_dict[k] = global_dict[k]
                    
                    client_model.load_state_dict(client_dict)
            
            # Train client model
            client_model, new_hyperparams = train_client(
                client_id=client_id,
                model=client_model,
                dataset=client_datasets[client_id],
                config=FedConfig,
                current_hyperparams=client_hyperparams[client_id]
            )
            
            # Store model and new hyperparameters
            client_models.append(client_model)
            updated_hyperparams.append(new_hyperparams)
        
        # Update client hyperparameters for next round
        client_hyperparams = updated_hyperparams
        
        # Aggregate models on the server
        global_model, agg_hyperparams = aggregate_models(global_model, client_models, client_hyperparams)
        
        print(f"\nAfter aggregation - Avg LR: {agg_hyperparams['learning_rate']:.2e}, Avg Rank: {agg_hyperparams['lora_rank']}")
        
        # Clear GPU memory before evaluation
        clear_gpu_memory()
        
        # Evaluate global model
        metrics = evaluate_model(global_model, val_dataloader)
        print(f"Round {round_num} - Val Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        # Save results
        results["rounds"].append(round_num)
        results["val_loss"].append(metrics["loss"])
        results["val_accuracy"].append(metrics["accuracy"])
        results["val_f1"].append(metrics["f1_score"])
        results["hyperparams"].append(client_hyperparams.copy())
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Save final model
    print("\nSaving final model...")
    global_model.save_pretrained("final_federated_model")
    print("Federated Learning completed! Final model saved.")
    
    return global_model, results

# Run the federated learning process
final_model, training_results = run_federated_learning()

# Visualize results
def plot_results(results):
    """Plot training metrics and hyperparameter evolution"""
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot validation loss
    axs[0, 0].plot(results["rounds"], results["val_loss"], 'o-', color='b')
    axs[0, 0].set_title('Validation Loss')
    axs[0, 0].set_xlabel('Round')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    # Plot validation accuracy
    axs[0, 1].plot(results["rounds"], results["val_accuracy"], 'o-', color='g')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Round')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True)
    
    # Plot learning rate evolution for each client
    for client_id in range(FedConfig.num_clients):
        lr_values = [results["hyperparams"][r][client_id]["learning_rate"] for r in range(len(results["rounds"]))]
        axs[1, 0].plot(results["rounds"], lr_values, 'o-', label=f'Client {client_id}')
    
    axs[1, 0].set_title('Learning Rate Evolution')
    axs[1, 0].set_xlabel('Round')
    axs[1, 0].set_ylabel('Learning Rate')
    axs[1, 0].set_yscale('log')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot LoRA rank evolution for each client
    for client_id in range(FedConfig.num_clients):
        rank_values = [results["hyperparams"][r][client_id]["lora_rank"] for r in range(len(results["rounds"]))]
        axs[1, 1].plot(results["rounds"], rank_values, 'o-', label=f'Client {client_id}')
    
    axs[1, 1].set_title('LoRA Rank Evolution')
    axs[1, 1].set_xlabel('Round')
    axs[1, 1].set_ylabel('LoRA Rank')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png')
    plt.show()

# Plot results
plot_results(training_results)

# Final evaluation on both validation sets
print("\nFinal evaluation on validation matched dataset:")
final_metrics_matched = evaluate_model(final_model, val_dataloader)
print(f"Final model - Val Loss: {final_metrics_matched['loss']:.4f}, Accuracy: {final_metrics_matched['accuracy']:.4f}, F1: {final_metrics_matched['f1_score']:.4f}")

# Create dataloader for mismatched validation set
val_mismatched_dataloader = DataLoader(
    val_mismatched_dataset,
    batch_size=FedConfig.val_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

print("\nFinal evaluation on validation mismatched dataset:")
final_metrics_mismatched = evaluate_model(final_model, val_mismatched_dataloader)
print(f"Final model - Val Loss: {final_metrics_mismatched['loss']:.4f}, Accuracy: {final_metrics_mismatched['accuracy']:.4f}, F1: {final_metrics_mismatched['f1_score']:.4f}")

# Save results to JSON
results_output = {
    "config": {
        "num_clients": FedConfig.num_clients,
        "communication_rounds": FedConfig.communication_rounds,
        "local_epochs": FedConfig.local_epochs,
        "batch_size": FedConfig.batch_size,
        "model_name": FedConfig.model_name,
        "dirichlet_alpha": FedConfig.dirichlet_alpha,
        "initial_hyperparams": {
            "learning_rate": FedConfig.init_learning_rate,
            "lora_rank": FedConfig.init_lora_rank
        }
    },
    "training_results": training_results,
    "final_metrics": {
        "matched": final_metrics_matched,
        "mismatched": final_metrics_mismatched
    },
    "client_data_distribution": client_stats
}

with open('federated_learning_results.json', 'w') as f:
    json.dump(results_output, f, indent=2)

print("\nResults saved to 'federated_learning_results.json'")