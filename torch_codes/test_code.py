import torch
import torch.nn as nn
from torch.optim.adam import Adam
from support.logger import Logger

import dataset
import module


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = module.MyModule().to(device=device)

sample = 0
correct = 0

model.load_state_dict(torch.load("trained_models/clap_train_LSTM_2.pth", map_location=device))

for data, label in dataset.test_dataloader:
    with torch.no_grad():
        pred, _, __ = model(data.type(torch.float32).to(device))

        correct += torch.sum(torch.argmax(pred,dim=1) == label.to(device))
        sample += pred.shape[0]

print(f"{correct} out of {sample}")
