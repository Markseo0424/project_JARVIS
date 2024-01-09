import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT
import os
import random
import matplotlib.pyplot as plt

dataset_size = 8000

true_path_dir = '../sample_wav/true_wav'
false_path_dir = '../sample_wav/false_wav'
expand_path_dir = '../sample_wav/expanded_true_wav'

real_true_file_list = os.listdir(true_path_dir)
true_file_list = os.listdir(expand_path_dir)
false_file_list = os.listdir(false_path_dir)

i = 0

true_audio = []
false_audio = []
for file in true_file_list:
    x, sr = torchaudio.load(expand_path_dir + '/' + file)

    if x.shape[1] <= 320000:
        true_audio.append(x)

for file in real_true_file_list:
    x, sr = torchaudio.load(true_path_dir + '/' + file)

    if x.shape[1] <= 320000:
        true_audio.append(x)

for file in false_file_list:
    x, sr = torchaudio.load(false_path_dir + '/' + file)

    if x.shape[1] <= 320000:
        false_audio.append(x)

print(len(true_audio), len(false_audio))

dataset_lst = []
max_len = 0

for i in range(dataset_size):
    if random.random() < 0.5:
        label = 1
        data = true_audio[random.randint(0, len(true_audio) - 1)]
    else:
        label = 0
        data = false_audio[random.randint(0, len(false_audio) - 1)]

    if max_len < data.shape[1]:
        max_len = data.shape[1]

    dataset_lst.append((data, label))

print(max_len, len(dataset_lst))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mel_spectrogram = nn.Sequential(
    AT.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80
    ),
    AT.AmplitudeToDB()
).to(device)

dummy = mel_spectrogram(torch.zeros((max_len,), device=device))
print(dummy.shape)

data = torch.ones((8000, dummy.shape[1], dummy.shape[0])) * -100
label = torch.zeros((8000, 1))

print(data.shape, label.shape)

for i, (d, l) in enumerate(dataset_lst):
    spec = mel_spectrogram(d.to(device))
    data[i][-spec.shape[2]:] = spec[0].T
    label[i] = l


class ClapDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.data[i], self.label[i][0]


train_dataset = ClapDataset(data[:-1000], label[:-1000])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ClapDataset(data[-1000:], label[-1000:])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == '__main__':
    for data, label in train_dataloader:
        print(data.shape, label.shape)
        _,ax = plt.subplots(4,4)
        for i in range(4):
            for j in range(4):
                ax[i][j].pcolor(data[4*i+j].T)
                ax[i][j].set_title(label[4*i+j,0])
        break
    plt.show()