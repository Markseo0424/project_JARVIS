from gpt_api.stream_gpt import *
from gpt_api.call_gpt import *

class GPTChat:
    def __init__(self, memory=0):
        self.memory = memory
        self.messages = []

    def query(self, query, vanilla = False):
        self.messages.append({"role": "user", "content": query})
        if vanilla:
            value = call_vanilla_gpt(messages=self.messages[-self.memory:])
        else:
            value = call_gpt(messages=self.messages[-self.memory:])
        self.messages.append({"role": "assistant", "content": value})

        return value

    def query_stream(self, query, onEndSentence = None,vanilla = False):
        self.messages.append({"role": "user", "content": query})
        if vanilla:
            value = stream_vanilla_gpt(messages=self.messages[-self.memory:], onEndSentence=onEndSentence)
        else:
            value = stream_gpt(messages=self.messages[-self.memory:], onEndSentence=onEndSentence)
        self.messages.append({"role": "assistant", "content": value})

        return value
    def forget(self):
        if 0 < self.memory < len(self.messages):
            self.messages = self.messages[-self.memory:]

    def clear_history(self):
        self.messages = []