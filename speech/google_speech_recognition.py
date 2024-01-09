import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import winsound

rate = 44100

Recognizer = sr.Recognizer()  # 인스턴스 생성
mic = sr.Microphone(sample_rate=rate)

def speech_to_text(filename):
    with mic as source:  # 안녕~이라고 말하면
        winsound.PlaySound("./sound_asset/record_start.wav", winsound.SND_FILENAME)
        print("start speech : ")
        audio = Recognizer.listen(source)
        winsound.PlaySound("./sound_asset/record_done.wav", winsound.SND_FILENAME)
    try:
        data = Recognizer.recognize_google(audio, language="ko")
    except:
        data = "이해하지 못했음"

    audio_data = audio.get_wav_data()
    data_s16 = np.frombuffer(audio_data, dtype=np.int16, count=len(audio_data)//2, offset=0)

    float_data = data_s16.astype(np.float32, order='C') / 32768.0

    write(filename, rate, data_s16)

    return data, float_data


#print(float_data)
#print(data)  # 안녕 출력
