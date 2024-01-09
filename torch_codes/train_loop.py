import torch
import torch.nn as nn
from torch.optim.adam import Adam

import dataset
import module


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = module.MyModule().to(device=device)

lr = 2e-5
epoch = 100

optimizer = Adam(model.parameters(), lr=lr)

for i in range(epoch):
    j = 0
    for data, label in dataset.train_dataloader:
        optimizer.zero_grad()

        h0 = torch.zeros((5, data.shape[0], 128),device=device)
        pred, _, __ = model(data.type(torch.float32).to(device), h0)
        #print(pred.shape)
        #print(pred, label)
        loss = nn.CrossEntropyLoss()(pred, label.type(torch.long).to(device))
        loss.backward()
        optimizer.step()

        print(f"epoch : {i+1}, step {j}, loss : {loss.item()}", end='\r', flush=True)
        j += 1

    print()

sample = 0
correct = 0

for data, label in dataset.test_dataloader:
    with torch.no_grad():

        h0 = torch.zeros((5, data.shape[0], 128),device=device)
        pred = model(data.type(torch.float32).to(device), h0)

        correct += torch.sum(torch.argmax(pred,dim=1) == label.to(device))
        sample += pred.shape[0]

print(f"{correct} out of {sample}")

torch.save(model.state_dict(), "trained_models/clap_train_LSTM.pth")