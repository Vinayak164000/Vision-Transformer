import random
import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.optim as Optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from dataset import BirdsDataset
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count()
data_path = Path("C:\\Users\\hp\\OneDrive\\Desktop\\computervision\\archive(3)")
train_dir = data_path / 'train'
test_dir = data_path / 'test'
train_data = pd.read_csv('Training_set.csv')
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
num_classes = len(label_encoder.classes_)
from ViT import VIT
from train import Trainer


if __name__ == "__main__":

    # Load data and create data loaders
    data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_data = BirdsDataset(dataframe= train_data, data_dir= train_dir, transform= data_transform)
    test_data = BirdsDataset(dataframe= test_data, data_dir = train_dir, transform=data_transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle=True,
        drop_last=len(train_data) % BATCH_SIZE != 0,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        num_workers=12,
        shuffle=False,
        drop_last=len(test_data) % BATCH_SIZE != 0,
    )
    model = VIT(num_class=len(label_encoder.classes_))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    #


    # Create trainer and train the model
    trainer = Trainer(model, train_loader, test_loader, optimizer, loss_fn=loss_fn)

    # torch.save(model.state_dict(),'C:\\Users\\hp\\OneDrive\\Desktop\\computervision\\archive(3)\\model_architecture.pth')