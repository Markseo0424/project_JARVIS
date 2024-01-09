import sounddevice as sd
import numpy as np

def print_volume(indata, frames, time, status):
    print(indata.shape)


with sd.InputStream(samplerate=16000, blocksize=400, callback=print_volume):
    sd.sleep(2000)
