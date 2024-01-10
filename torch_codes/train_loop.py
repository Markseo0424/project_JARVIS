import torch
import torch.nn as nn
from torch.optim.adam import Adam
from support.logger import Logger

import dataset
import module


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = module.MyModule().to(device=device)

lr = 1e-5
epoch = 200

optimizer = Adam(model.parameters(), lr=lr)
logger = Logger()

for i in range(epoch):
    j = 0
    for data, label in dataset.train_dataloader:
        optimizer.zero_grad()

        pred, _, __ = model(data.type(torch.float32).to(device))
        #print(pred.shape)
        #print(pred, label)
        loss = nn.CrossEntropyLoss()(pred, label.type(torch.long).to(device))
        loss.backward()
        optimizer.step()

        logger.print(f"epoch : {i+1}, step {j}, loss : {loss.item()}", progress=(j + 1)/(len(dataset.train_dataset)//32))
        j += 1

    print()

torch.save(model.state_dict(), "trained_models/clap_train_LSTM_2.pth")

sample = 0
correct = 0

model.load_state_dict(torch.load("trained_models/clap_train_LSTM_2.pth", map_location=device))

for data, label in dataset.test_dataloader:
    with torch.no_grad():
        pred, _, __ = model(data.type(torch.float32).to(device))

        correct += torch.sum(torch.argmax(pred,dim=1) == label.to(device))
        sample += pred.shape[0]

print(f"{correct} out of {sample}")
