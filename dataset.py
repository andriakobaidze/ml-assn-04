import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.emotions = df['emotion'].values
        self.pixels = df['pixels'].values
        self.transform = transform

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        pixel_values = np.array(self.pixels[idx].split(), dtype=np.float32)
        image = pixel_values.reshape(48, 48)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.emotions[idx], dtype=torch.long)
        return image, label


def get_transforms(augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(48, padding=4),
        ])
    else:
        train_transform = None

    return train_transform


def get_dataloaders(train_csv_path, test_csv_path, val_split=0.2, batch_size=64, augment=True):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    val_size = int(len(train_df) * val_split)
    train_size = len(train_df) - val_size

    train_df_split = train_df.iloc[:train_size].reset_index(drop=True)
    val_df_split = train_df.iloc[train_size:].reset_index(drop=True)

    train_transform = get_transforms(augment=augment)

    train_dataset = FERDataset(train_df_split, transform=train_transform)
    val_dataset = FERDataset(val_df_split, transform=None)
    test_dataset = FERDataset(test_df, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader