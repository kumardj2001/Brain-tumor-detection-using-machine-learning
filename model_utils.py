import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import pandas as pd

class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, height=128, width=128):
        self.no_class = glob(data_dir + '/no/*')
        self.yes_class = glob(data_dir + '/yes/*')
        self.height, self.width = height, width

        labels = [0]*len(self.no_class) + [1]*len(self.yes_class)
        image_links = self.no_class + self.yes_class
        self.dataframe = pd.DataFrame({"image": image_links, "labels": labels})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image"]).convert("L").resize((self.width, self.height))
        image = np.asarray(image).reshape(1, self.height, self.width)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(row["labels"], dtype=torch.long)

class BrainTumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 32, kernel_size=2)
        )
        self.linear1 = nn.Linear(62, 128)
        self.linear2 = nn.Linear(128, 64)
        self.flat = nn.Flatten(1)
        self.linear3 = nn.Linear(126976, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.flat(x)
        x = self.linear3(x)
        return x
