import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=5):
        super(MyModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.RNN = nn.RNN(input_size=80, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.LSTM = nn.LSTM(input_size=80, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.tanh1 = nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(x)

        if c0 is None:
            c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(x)

        # x, hn = self.RNN(x, h0)
        x, (hn, cn) = self.LSTM(x, (h0, c0))
        if len(x.shape) == 2:
            x = x[-1]
        else :
            x = x[:, -1]
        x = self.tanh1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x, hn, cn
