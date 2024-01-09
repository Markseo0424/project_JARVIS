import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
import os
import random

true_path_dir = '../sample_wav/true_wav'
false_path_dir = '../sample_wav/false_wav'
expand_path_dir = '../sample_wav/expanded_true'

true_file_list = os.listdir(true_path_dir)
false_file_list = os.listdir(false_path_dir)

i = 0

true_audio = []
false_audio = []
for file in true_file_list:
    x, sr = torchaudio.load(true_path_dir + '/' + file)
    true_audio.append(x)

for file in false_file_list:
    x, sr = torchaudio.load(false_path_dir + '/' + file)

    if x.shape[1] > 80000:
        false_audio.append(x)

print(len(true_audio), len(false_audio))

index = 0
for f_audio in false_audio:
    for t_audio in true_audio:
        if f_audio.shape[1] >= t_audio.shape[1]:
            shift_rand = random.randint(0, f_audio.shape[1] - t_audio.shape[1] - 1)
            e_audio = f_audio.clone()
            e_audio[:,shift_rand : shift_rand + t_audio.shape[1]] += t_audio
            torchaudio.save(expand_path_dir+'/test'+str(index).zfill(5)+'.wav', e_audio, 16000)
            index += 1