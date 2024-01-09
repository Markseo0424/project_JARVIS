import whisper
import torch


model = whisper.load_model("base")
result = model.transcribe("../chat_wav/test000.wav",language="Korean")
print(result["text"])