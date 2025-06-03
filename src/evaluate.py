# src/evaluate.py

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import get_data_loaders
from src.model_cnn import get_model

def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if images.shape[1] == 3:
                images = images[:, 0:1, :, :]

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("🧪 Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.show()


def run_evaluation():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'outputs/best_model.pth'
    DATA_DIR = 'data/chest_xray'

    _, _, test_loader = get_data_loaders(DATA_DIR, batch_size=32, img_size=224)
    model = get_model(model_name='efficientnet', pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    evaluate(model, test_loader, DEVICE)

if __name__ == '__main__':
    run_evaluation()
