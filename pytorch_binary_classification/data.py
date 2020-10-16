import torch
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Custom_Dataset(Dataset):

    def __init__(self, x, y, transform=None):
        self.x = x[:, :33]
        self.y = y
        self.file_names = x[:, 33]
        self.transform = transform


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y,dtype=torch.float32), idx, self.file_names[idx]

def get_loader(args):
    df = pd.read_pickle('./DGA_PP_DB.pickle')

    x_train = df['x_train']
    y_train = df['y_train']
    x_test = df['x_test']
    y_test = df['y_test']

    train_dataset = Custom_Dataset(x_train, y_train)
    test_dataset = Custom_Dataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4)

    return train_loader, test_loader