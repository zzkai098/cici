"""Conversation state and context management.

TODO (roadmap 3): persistence to .cici/sessions/, context window management
(what to keep, what to summarise, what to drop), @file mention injection as
<document id="..."> blocks (ported from cici_101/cli_project/core/cli_chat.py).
"""


class Session:
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
        self.messages = []

    def add_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, message):
        content = getattr(message, "content", message)
        self.messages.append({"role": "assistant", "content": content})

    def __len__(self):
        return len(self.messages)
