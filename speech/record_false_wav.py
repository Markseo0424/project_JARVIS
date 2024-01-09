import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import random
import os

sound = np.zeros(10 * 16000)
index = 0

dirname = '../sample_wav/false_wav'

file_list = os.listdir(dirname)
file_num = len(file_list)


def record_sound(indata, frames, time, status):
    global index, sound, file_num

    if random.random() < 0.005:
        sound_int = sound[:index * 800] * 32768
        sound_int = sound_int.astype(np.int16)

        write(dirname + f'/test{str(file_num).zfill(5)}.wav', 16000, sound_int)
        index = 0
        file_num += 1

    try:
        sound[index * 800: (index + 1) * 800] = indata[:, 0]
    except:
        return
    index += 1


with sd.InputStream(samplerate=16000, blocksize=800, callback=record_sound):
    while True:
        sd.wait(10000)

    # 28006 -21436
