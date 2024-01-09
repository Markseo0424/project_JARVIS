from gtts import gTTS
import winsound
import torchaudio


def play_text(text, filename):
    tts = gTTS(text=text, lang='ko')
    tts.save(filename)
    x, sr = torchaudio.load(filename)
    torchaudio.save(filename, x, sr)

    winsound.PlaySound(filename, winsound.SND_FILENAME)


def speak(texts, filename='./chat_wav/temp.wav'):
    #print(texts)
    if len(texts) < 1:
        return

    if type(texts) == list:
        for text in texts:
            try:
                play_text(text, filename)
            except:
                continue

    elif type(texts) == str:
        try:
            play_text(texts, filename)
        except:
            return


if __name__ == "__main__":
    speak("안녕 내이름은 빅스비야")
