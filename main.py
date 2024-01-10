import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
import time
import sounddevice as sd
import torch_codes.module as module
from gpt_api.gpt_chat import GPTChat
from speech.google_speech_recognition import speech_to_text
from speech.speak_text import speak
from task_modules.task_main import *
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
model.load_state_dict(torch.load("torch_codes/trained_models/clap_train_LSTM_2.pth", map_location=device))

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
        # print(pred)
        if pred[1] - pred[0] > 5:
            flag = 0


chat = GPTChat(memory=1)

predict_path_dir = './sample_wav/pred_true_wav'
file_list = os.listdir(predict_path_dir)

chat_index = 0
file_count = len(file_list)

print("===========[ START SERVICE ]==========")
while True:
    with sd.InputStream(samplerate=16000, blocksize=16000, callback=clap_predict):
        while flag:
            query, trigger = sch.check()
            if trigger:
                flag = False
                print("query : ", query)
            continue

    flag = True

    if not trigger:
        print("clap!")

        h0 = torch.zeros((5, 128), device=device)
        c0 = torch.zeros((5, 128), device=device)
        torchaudio.save(f'./sample_wav/pred_true_wav/test{str(file_count).zfill(5)}.wav',
                        history[:, -16000 * history_index:], 16000)
        file_count += 1
        history = torch.zeros((1, 160000))
        history_index = 0

        query, float_data = speech_to_text("./chat_wav/test" + str(chat_index).zfill(3) + ".wav")
        print("query : ", query)

        if query == "종료":
            speak("프로그램을 종료합니다.")
            break
        elif "취소" in query:
            speak("취소합니다.")
            continue
        elif query == "이해하지 못했음":
            speak("이해하지 못했습니다.")
            continue

    loop = True
    vanilla = False

    while loop:
        value = chat.query_stream(query, speak, vanilla)
        print("value : ", value)

        text, commands = split_command(value)
        # if text:
        #     speak(text, f"chat_wav/value{str(index).zfill(3)}.wav")

        try:
            query, loop = do_task(commands, query)
        except:
            vanilla = True
            loop = True

        if loop:
            print("loop query : ", query)

    chat_index += 1
