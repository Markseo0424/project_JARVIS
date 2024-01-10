import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT
import os
import random
import matplotlib.pyplot as plt

dataset_size = 8000

true_path_dirs = [
    '../sample_wav/true_wav',
    '../sample_wav/expanded_true_wav',
    '../sample_wav/pred_true_positive_wav'
]

false_path_dirs = [
    '../sample_wav/false_wav',
    '../sample_wav/pred_false_positive_wav'
]

true_weight = [1, 1, 1]
false_weight = [1, 1]

i = 0

print("importing audios..")
true_audios = []
false_audios = []
for true_path in true_path_dirs:
    true_audio = []
    true_file_list = os.listdir(true_path)
    for file in true_file_list:
        x, sr = torchaudio.load(true_path + '/' + file)

        if x.shape[1] <= 320000:
            true_audio.append(x)
    true_audios.append(true_audio)

for false_path in false_path_dirs:
    false_audio = []
    false_file_list = os.listdir(false_path)
    for file in false_file_list:
        x, sr = torchaudio.load(false_path + '/' + file)

        if x.shape[1] <= 320000:
            false_audio.append(x)
    false_audios.append(false_audio)


dataset_lst = []
max_len = 0

print("making long samples..")
# long samples
for i in range(dataset_size//2):
    if random.random() < 0.5:
        label = 1

        mult = random.randint(0, 9)

        audio_assemble = []

        for j in range(mult):
            choice = random.choices(list(range(len(false_audios))),weights=false_weight,k=1)[0]
            false_audio = false_audios[choice]
            audio_assemble.append(false_audio[random.randint(0, len(false_audio) - 1)])

        choice = random.choices(list(range(len(true_audios))),weights=true_weight,k=1)[0]
        true_audio = true_audios[choice]
        audio_assemble.append(true_audio[random.randint(0, len(true_audio) - 1)])

        data = torch.concat(audio_assemble, dim=1)

    else:
        label = 0

        mult = random.randint(1,10)

        audio_assemble = []

        for j in range(mult):
            choice = random.choices(list(range(len(false_audios))),weights=false_weight,k=1)[0]
            false_audio = false_audios[choice]
            audio_assemble.append(false_audio[random.randint(0, len(false_audio) - 1)])

        data = torch.concat(audio_assemble, dim=1)

    if max_len < data.shape[1]:
        max_len = data.shape[1]

    dataset_lst.append((data, label))


print("making simple samples..")
for i in range(dataset_size//2):
    if random.random() < 0.5:
        label = 1
        choice = random.choices(list(range(len(true_audios))),weights=true_weight,k=1)[0]
        true_audio = true_audios[choice]
        data = true_audio[random.randint(0, len(true_audio) - 1)]
    else:
        label = 0
        choice = random.choices(list(range(len(false_audios))),weights=false_weight,k=1)[0]
        false_audio = false_audios[choice]
        data = false_audio[random.randint(0, len(false_audio) - 1)]

    if max_len < data.shape[1]:
        max_len = data.shape[1]

    dataset_lst.append((data, label))

print(max_len, len(dataset_lst))

print("creating dataset..")
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

data = torch.ones((dataset_size, dummy.shape[1], dummy.shape[0])) * -100
label = torch.zeros((dataset_size, 1))

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


testset_size = dataset_size // 8

train_dataset = ClapDataset(data[:-testset_size], label[:-testset_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ClapDataset(data[-testset_size:], label[-testset_size:])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == '__main__':
    for data, label in train_dataloader:
        print(data.shape, label.shape)
        _, ax = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                ax[i][j].pcolor(data[4 * i + j].T)
                ax[i][j].set_title(label[4 * i + j].item())
        break
    plt.show()
