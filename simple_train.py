import os
import random
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from datasets import load_dataset, Features, ClassLabel, Image as HFImage
from PIL import Image
from tqdm.auto import tqdm # For progress bar
import matplotlib.pyplot as plt # For plotting
import math

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration ---
# IMPORTANT: Set CUDA_VISIBLE_DEVICES=X (e.g., 3) in your terminal
# BEFORE running this script. E.g., export CUDA_VISIBLE_DEVICES=3
DEVICE_ID = 0 # PyTorch will use cuda:0, which maps to the GPU specified by CUDA_VISIBLE_DEVICES
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda' and DEVICE.index is not None: # Further check if specific CUDA device is truly available
    try:
        torch.cuda.get_device_name(DEVICE.index)
    except AssertionError: # PyTorch raises AssertionError if device ordinal is invalid
        print(f"Warning: CUDA device cuda:{DEVICE.index} is not available or invalid. Falling back to CPU.")
        DEVICE = torch.device("cpu")
    except RuntimeError as e: # Catch other CUDA errors like "invalid device function" if drivers are an issue
        print(f"Warning: CUDA error ({e}) on device cuda:{DEVICE.index}. Falling back to CPU.")
        DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


TRAIN_FILE_PATH = "/home/ps/llm/segmentation/fl/data/train-00000-of-00001.parquet"
TEST_FILE_PATH = "/home/ps/llm/segmentation/fl/data/test-00000-of-00001.parquet"

NUM_CLIENTS = 10
CLIENTS_PER_ROUND = 2
NUM_ROUNDS = 20
LOCAL_EPOCHS = 3 # Transformers might need more or less, depends on complexity
BATCH_SIZE = 32  # May need to reduce if ViT is memory intensive

# GTSRB Specific Dataset and Model parameters
IMG_SIZE = 112
IMG_CHANNELS = 3
NUM_CLASSES = 43

# --- Vision Transformer Model Definition ---
class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=IMG_SIZE, patch_size=8, in_chans=IMG_CHANNELS, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # Use Conv2d for patch embedding: kernel_size and stride are the patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=8, in_chans=IMG_CHANNELS, num_classes=NUM_CLASSES,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation='relu', # or 'gelu'
            batch_first=True, # Important: (Batch, Seq, Feature)
            norm_first=True # Often recommended for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)
        x = self.norm(x)

        # Use CLS token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits

# --- Custom Strategy with TQDM (No changes needed from previous version) ---
class FedAvgWithTQDM(fl.server.strategy.FedAvg):
    def __init__(self, *args, num_rounds_total: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds_total = num_rounds_total
        self.pbar = tqdm(total=self.num_rounds_total, desc="Federated Rounds", unit="round", dynamic_ncols=True)

    def evaluate(self, server_round: int, parameters: fl.common.NDArrays) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        eval_results = super().evaluate(server_round, parameters)
        if eval_results is not None:
            loss, metrics = eval_results
            self.pbar.set_postfix(
                ordered_dict=OrderedDict(
                    [("loss", f"{loss:.4f}"), ("accuracy", f"{metrics.get('accuracy', 0.0):.4f}")]
                )
            )
        if self.pbar.n < server_round:
            self.pbar.update(server_round - self.pbar.n)
        return eval_results

    def close_pbar(self):
        if hasattr(self, 'pbar') and self.pbar:
            if self.pbar.n < self.num_rounds_total: # If simulation ended early
                self.pbar.update(self.num_rounds_total - self.pbar.n)
            self.pbar.refresh()
            self.pbar.close()

# --- Dataset Handling (No changes needed from previous version) ---
class ParquetImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image, label = item['image'], item['label']
        if not isinstance(image, Image.Image): raise TypeError(f"Expected PIL image, got {type(image)}")
        if image.mode != 'RGB': image = image.convert('RGB')
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def load_data():
    data_files = {'train': TRAIN_FILE_PATH, 'test': TEST_FILE_PATH}
    features = Features({'image': HFImage(), 'label': ClassLabel(num_classes=NUM_CLASSES)})
    try:
        raw_datasets = load_dataset("parquet", data_files=data_files, features=features)
    except Exception as e:
        print(f"Error loading dataset with specified features: {e}. Attempting to load without.")
        raw_datasets = load_dataset("parquet", data_files=data_files)
        max_label = max(max(raw_datasets['train']['label']), max(raw_datasets['test']['label']))
        if max_label + 1 != NUM_CLASSES: print(f"Warning: Max label ({max_label})+1 != NUM_CLASSES ({NUM_CLASSES}).")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])
    full_train_dataset = ParquetImageDataset(raw_datasets['train'], transform=transform)
    test_dataset = ParquetImageDataset(raw_datasets['test'], transform=transform)
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices) 
    len_per_client = len(full_train_dataset) // NUM_CLIENTS
    client_datasets = [Subset(full_train_dataset, indices[i*len_per_client : (i+1)*len_per_client if i < NUM_CLIENTS-1 else len(full_train_dataset)]) for i in range(NUM_CLIENTS)]
    client_trainloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False) for ds in client_datasets]
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    return client_trainloaders, testloader

# --- Flower Client Implementation (No changes needed from previous version) ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model: nn.Module, trainloader: DataLoader):
        self.cid = cid
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
    def get_parameters(self, config): return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        # Transformers might benefit from AdamW and a learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2) # Example
        criterion = nn.CrossEntropyLoss()
        epoch_loss, num_samples_total = 0.0, 0
        for _ in range(int(config.get("local_epochs", LOCAL_EPOCHS))):
            current_epoch_loss, current_epoch_samples = 0.0, 0
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                current_epoch_loss += loss.item() * images.size(0)
                current_epoch_samples += images.size(0)
            epoch_loss += current_epoch_loss
            num_samples_total = current_epoch_samples
        avg_loss_over_epochs = epoch_loss / (num_samples_total * int(config.get("local_epochs", LOCAL_EPOCHS))) if num_samples_total > 0 else float('inf')
        return self.get_parameters(config={}), num_samples_total, {"loss": avg_loss_over_epochs}
    def evaluate(self, parameters, config): # Mostly for completeness, server eval is primary
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.trainloader: # Placeholder: use a valloader
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss / total if total > 0 else float('inf')
        return avg_loss, total, {"accuracy": accuracy, "cid": self.cid}

# --- Server-Side Logic & Simulation ---
def client_fn_simulation(cid: str, client_trainloaders_list: List[DataLoader]) -> FlowerClient:
    # Instantiate the VisionTransformer model
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=8, # (48 // 8 = 6 patches per dim -> 36 patches)
        in_chans=IMG_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=192, # Embedding dimension (e.g., 192, 256) - keep it modest for GTSRB
        depth=3,       # Number of Transformer encoder layers (e.g., 3, 4)
        num_heads=3,   # Number of attention heads (must be a divisor of embed_dim, e.g., 3, 4, 6)
        mlp_ratio=4.,
        dropout_rate=0.1
    )
    trainloader_for_client = client_trainloaders_list[int(cid)]
    return FlowerClient(cid, model, trainloader_for_client)

def fit_config(server_round: int) -> Dict:
    return {"server_round": server_round, "local_epochs": LOCAL_EPOCHS}

def evaluate_fn_centralized(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar], testloader_global: DataLoader) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    # Instantiate the VisionTransformer model
    model = VisionTransformer(
        img_size=IMG_SIZE, patch_size=8, in_chans=IMG_CHANNELS, num_classes=NUM_CLASSES,
        embed_dim=192, depth=3, num_heads=3, mlp_ratio=4.
    ) # Ensure consistency with client_fn_simulation
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in testloader_global:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0: return 0.0, {"accuracy": 0.0}
    avg_loss = loss / total
    accuracy = correct / total
    return avg_loss, {"accuracy": accuracy}

def main():
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if DEVICE.type == 'cuda':
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        try:
            current_device_id = torch.cuda.current_device() # This will be 0 if CUDA_VISIBLE_DEVICES restricted it
            print(f"PyTorch current CUDA device (as seen by script): cuda:{current_device_id} ({torch.cuda.get_device_name(current_device_id)})")
        except Exception as e:
            print(f"Could not get current CUDA device details: {e}")


    client_trainloaders, testloader = load_data()
    
    strategy = FedAvgWithTQDM(
        num_rounds_total=NUM_ROUNDS,
        fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS, 
        min_fit_clients=CLIENTS_PER_ROUND,          
        min_available_clients=NUM_CLIENTS,        
        fraction_evaluate=0.0, 
        min_evaluate_clients=0,
        evaluate_fn=lambda sr, p, c: evaluate_fn_centralized(sr, p, c, testloader),
        on_fit_config_fn=fit_config,
    )

    print(f"Starting GTSRB simulation with Vision Transformer: {NUM_CLIENTS} clients, {CLIENTS_PER_ROUND} per round for {NUM_ROUNDS} rounds.")
    
    history = None
    try:
        # Adjust client_gpu_resource based on your ViT's memory footprint and your GPU.
        # If your ViT is large, you might only be able to fit one on a GPU at a time.
        client_gpu_resource = 0.0 # Default: Ray actor doesn't request dedicated GPU from Ray, uses process's DEVICE
        if DEVICE.type == 'cuda':
            # client_gpu_resource = 0.45 # Example for sharing a GPU
            client_gpu_resource = 1.0 # If each client actor should request a full GPU from Ray (if available)

        print(f"Using client_resources: num_cpus=2, num_gpus={client_gpu_resource}")
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn_simulation(cid, client_trainloaders),
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": client_gpu_resource},
        )
    finally:
        strategy.close_pbar()

    print("\nSimulation finished.")
    if history and history.metrics_centralized:
        print("Centralized metrics history (accuracy):", history.metrics_centralized.get("accuracy"))
        print("Centralized losses history:", history.losses_centralized)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5)) # Create a figure and a set of subplots

        if history.losses_centralized:
            rounds_loss, values_loss = zip(*history.losses_centralized)
            axs[0].plot(rounds_loss, values_loss, marker='o', linestyle='-')
            axs[0].set_title("Server-Side Loss per Round")
            axs[0].set_xlabel("Round")
            axs[0].set_ylabel("Loss")
            axs[0].grid(True)
            axs[0].set_xticks(rounds_loss)
        else:
            axs[0].set_title("No centralized losses recorded")
            print("No centralized losses recorded to plot.")

        if "accuracy" in history.metrics_centralized and history.metrics_centralized["accuracy"]:
            rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
            axs[1].plot(rounds_acc, values_acc, marker='x', linestyle='-')
            axs[1].set_title("Server-Side Accuracy per Round")
            axs[1].set_xlabel("Round")
            axs[1].set_ylabel("Accuracy")
            axs[1].grid(True)
            axs[1].set_xticks(rounds_acc)
        else:
            axs[1].set_title("No centralized accuracy recorded")
            print("No centralized accuracy recorded to plot.")
        
        if history.losses_centralized or ("accuracy" in history.metrics_centralized and history.metrics_centralized["accuracy"]):
            plt.tight_layout()
            plt.show()
        else:
            print("No sufficient metrics recorded to plot.")

    else:
        print("No history or centralized metrics recorded from simulation.")

if __name__ == "__main__":
    # REMEMBER to set CUDA_VISIBLE_DEVICES in your shell if you want to target a specific GPU.
    # e.g., export CUDA_VISIBLE_DEVICES=3
    main()