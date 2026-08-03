"""
The code below trains a simple neural network on the Iris datraset.

First, the dataset is split into training and testing data.
"""
from numpy import unique
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from tqdm import tqdm

from sklearn.datasets import load_iris

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

#torch.manual_seed(42)

##########################
# 1. DATA
##########################
X, y = load_iris(return_X_y=True)
n_classes = len(unique(y))

ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
ds_train, ds_test = random_split(ds, [0.8, 0.2])
dl_train = DataLoader(ds_train, batch_size=8, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=8, shuffle=False)

N,F = X.shape
model = nn.Sequential(nn.Linear(F, 16), nn.ReLU(), nn.Linear(16, n_classes)) # There is no activation on the last layer because it is included in the CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epoch in range(1):
    for xb, yb in dl_train:
        y_pred = model(xb)
        loss = loss_fn(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
correct=0
total=0
model.eval()
with torch.no_grad():
    for xb, yb in dl_test:
        y_pred = model(xb)
        correct += (y_pred.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)
print(correct/total)
