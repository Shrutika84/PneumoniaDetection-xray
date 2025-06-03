import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class RGBDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path).convert("RGB")  # 🔥 force RGB
        if self.transform is not None:
            image = self.transform(image)
        print(f"[DEBUG] Loaded image shape: {image.shape}")  # Should be [3, 224, 224]
        return image, label

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # From RGB: (H, W, C) → (3, H, W)
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = RGBDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset   = RGBDataset(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset  = RGBDataset(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
