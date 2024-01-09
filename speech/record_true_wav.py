import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import winsound

rate = 16000

Recognizer = sr.Recognizer()  # 인스턴스 생성
mic = sr.Microphone(sample_rate=rate)

dirname = '../sample_wav/true_wav'

file_list = os.listdir(dirname)
file_num = len(file_list)


while True:
    with mic as source:  # 안녕~이라고 말하면
        winsound.PlaySound("../sound/record_start.wav", winsound.SND_FILENAME)
        print("start speech : ")
        audio = Recognizer.listen(source)
        winsound.PlaySound("../sound/record_done.wav", winsound.SND_FILENAME)

    if not input("save?"):
        audio_data = audio.get_wav_data()
        data_s16 = np.frombuffer(audio_data, dtype=np.int16, count=len(audio_data) // 2, offset=0)

        print(filename + str(file_num).zfill(5) + ".wav")
        write(filename + str(file_num).zfill(5) + ".wav", rate, data_s16[2000:])
        file_num += 1