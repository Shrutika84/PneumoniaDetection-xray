# main.py

import os
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_loader import get_data_loaders
from model_cnn import get_model
from train import train_model




def main():
    # --- Configs ---
    DATA_DIR = 'data/chest_xray'  
    MODEL_TYPE = 'efficientnet' 
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 10
    LR = 1e-4
    SAVE_PATH = 'outputs/best_model.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Prepare Data ---
    print(" Loading data...")
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)

    # --- Build Model ---
    print(f"🧠 Initializing {MODEL_TYPE} model...")
    model = get_model(model_name=MODEL_TYPE, pretrained=True)

    # --- Train ---
    print("🚀 Starting training...")
    os.makedirs('outputs', exist_ok=True)
    train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR, SAVE_PATH)
    print("✅ Training complete.")

if __name__ == '__main__':
    main()
