import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)


def call_gpt(query=None, messages=None):
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
        stop=['\r']
    )

    return response.choices[0].message.content


def call_vanilla_gpt(query=None, messages=None):
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
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['\r']
    )

    return response.choices[0].message.content
