## this code predicts real time audio whether there is clap or not.
## if there is clap, saves audio to pred_true directory.

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
import time
import sounddevice as sd
import module
import winsound
import os

flag = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

mel_spectrogram = nn.Sequential(
    AT.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80
    ),
    AT.AmplitudeToDB()
)

model = module.MyModule().to(device)
model.load_state_dict(torch.load("trained_models/clap_train_LSTM.pth", map_location=device))

h0 = torch.zeros((5, 128), device=device)
c0 = torch.zeros((5, 128), device=device)

history = torch.zeros((1, 160000))
history_index = 0

def clap_predict(indata, frames, t, status):
    # print(indata.shape)
    global h0, c0, flag, history, history_index

    hist_clone = history[0, 16000:].clone()
    history[0, :-16000] = hist_clone
    history[0, -16000:] = torch.tensor(indata[:, 0])
    if history_index < 10:
        history_index += 1

    x = mel_spectrogram(torch.tensor(indata[:, 0]))
    # print(x.T.shape)
    with torch.no_grad():
        pred, h0, c0 = model(x.T.to(device), h0, c0)
        print(pred)
        if pred[1] - pred[0] > 5:
            flag = 0


predict_path_dir = '../sample_wav/pred_true_wav'
file_list = os.listdir(predict_path_dir)

file_num = len(file_list)

while True:
    with sd.InputStream(samplerate=16000, blocksize=16000, callback=clap_predict):
        while flag:
            continue

    print("clap!")
    winsound.PlaySound("../sound_asset/record_done.wav", winsound.SND_FILENAME)
    h0 = torch.zeros((5, 128), device=device)
    c0 = torch.zeros((5, 128), device=device)
    flag = True
    torchaudio.save(f'../sample_wav/pred_true_wav/test{str(file_num).zfill(5)}.wav', history[:,-16000 * history_index:], 16000)
    file_num += 1
    history = torch.zeros((1, 160000))
    history_index = 0
