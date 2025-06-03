import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, save_path='outputs/best_model.pth'):
    """
    Trains the model and saves the best weights based on validation accuracy.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]')

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # --- Skip if somehow 1-channel (safety check) ---
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # --- Validation ---
        val_acc = evaluate_model(model, val_loader, device)
        print(f'\n✅ Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f' Best model saved with accuracy: {val_acc:.2f}%\n')


def evaluate_model(model, data_loader, device):
    """
    Evaluates model accuracy on the given data loader.
    Converts grayscale to RGB if needed.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Force 3-channel 
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total
