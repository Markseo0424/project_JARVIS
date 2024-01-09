from gpt_api.call_gpt import call_gpt
from gpt_api.gpt_chat import GPTChat
from speech.google_speech_recognition import speech_to_text

chat = GPTChat()

index = 0
while True:
    query, float_data = speech_to_text("./chat_wav/test" + str(index).zfill(3) + ".wav")
    print("query : ", query)

    if query == "종료":
        break
    if query == "이해하지 못했음":
        continue

    value = chat.query(query)
    print("value : ", value)

    index += 1
