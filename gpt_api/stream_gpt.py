import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)


def stream_gpt(query=None, messages=None, onEndSentence=None):
    if messages is None:
        if query is None:
            print("error : no query")
            return ""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::8f5yBzSN",
        messages=messages,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['\r'],
        stream=True
    )

    response_string = ""
    sentence_index = 0

    for chunk in response:
        chunk_string = chunk.choices[0].delta.content

        if chunk_string is None:
            continue

        response_string += chunk_string

        if "!module " in response_string :
            continue

        if onEndSentence is not None and ("." in chunk_string or "?" in chunk_string or "!" in chunk_string):
            split = splitter(response_string)
            sentence_list = split[:-1]
            onEndSentence(sentence_list[sentence_index:])
            sentence_index = len(sentence_list)

    return response_string

def stream_vanilla_gpt(query=None, messages=None, onEndSentence=None):
    if messages is None:
        if query is None:
            print("error : no query")
            return ""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['\r'],
        stream=True
    )

    response_string = ""
    sentence_index = 0

    for chunk in response:
        chunk_string = chunk.choices[0].delta.content
        if chunk_string is None:
            continue
        response_string += chunk_string
        if onEndSentence is not None and ("." in chunk_string or "?" in chunk_string or "!" in chunk_string):
            split = splitter(response_string)
            sentence_list = split[:-1]
            onEndSentence(sentence_list[sentence_index:])
            sentence_index = len(sentence_list)

    return response_string

def splitter(sentence):
    res = []
    delimeters = [".", "?", "!"]

    s = ""
    for i in range(len(sentence)):
        c = sentence[i]

        if c in delimeters:
            res.append(s.strip() + c)
            s = ""
        else:
            s += c

    res.append(s)

    return res

if __name__ == "__main__" :
    print(stream_vanilla_gpt("한글의 기원을 말해봐", onEndSentence=print))